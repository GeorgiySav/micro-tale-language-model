from dataclasses import dataclass
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F

import tiktoken

@dataclass
class GPTConfig:
  block_size: int = 1024
  vocab_size: int = 50257
  num_layers: int = 12
  num_heads: int = 12
  num_embed: int = 768
  dropout: int = 0.2

class CasualSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.num_embed % config.num_heads == 0
    # key, query, value projections for all heads, but in a batch
    self.c_attention = nn.Linear(config.num_embed, 3*config.num_embed)
    # output projection
    self.c_projection = nn.Linear(config.num_embed, config.num_embed)
    self.c_projection.GPT_SCALE_INIT = 1.0
    # regularization
    self.num_heads = config.num_heads
    self.num_embed = config.num_embed
    self.dropout = config.dropout

  def forward(self, x):
    B, T, C = x.size() # batch size, sequence length, embedding fimentionality

    # calculate query, key and values for all heads in batch and move head forward to be the batch
    # nh is number of heads, hs is head size, and C (number of channels) = nh * ns
    qkv = self.c_attention(x)
    q, k, v = qkv.split(self.num_embed, dim=2)
    q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
    k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemable all head outputs side by side
    
    return self.c_projection(y)


class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.num_embed, 4 * config.num_embed)
    self.gelu = nn.GELU()
    self.c_proj = nn.Linear(4 * config.num_embed, config.num_embed)
    self.c_proj.GPT_SCALE_INIT = 1.0
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    x = self.dropout(x)
    return x


class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.num_embed)
    self.attention = CasualSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.num_embed)
    self.mlp = MLP(config)
  
  def forward(self, x):
    x = x + self.attention(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x
  

class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.transformer = nn.ModuleDict(dict(
      token_embedding = nn.Embedding(config.vocab_size, config.num_embed),
      position_embedding = nn.Embedding(config.block_size, config.num_embed),
      blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
      ln_f = nn.LayerNorm(config.num_embed)
    ))
    self.lm_head = nn.Linear(config.num_embed, config.vocab_size, bias=False)

    # weight sharing scheme
    self.transformer.token_embedding.weight = self.lm_head.weight

    # init params
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      std = 0.02
      if hasattr(module, 'GPT_SCALE_INIT'):
        std *= (2 * self.config.num_layers) ** -0.5
      torch.nn.init.normal_(module.weight, mean=0.0, std=std)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    # idx is of shape (B, T)
    B, T = idx.size()
    assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
    # forward the token and position embedding
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
    tok_embed = self.transformer.token_embedding(idx) # (B, T, num_embed)
    pos_embed = self.transformer.position_embedding(pos) # (T, num_embed)
    x = tok_embed + pos_embed
    # forward the blocks of the transformer
    for block in self.transformer.blocks:
      x = block(x)
    # forward the final layer norm and the classifier
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x) # (B, T, vocab_size)
    
    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    return logits, loss
  
  def configure_optimizers(self, weight_decay, learning_rate, device):
    # start with all of the candidate parameters that require grad
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optimizer groups. Any parameters that is 2d will be weight decayed, otherwise no
    # all weight tensors in matmuls and embeddings decay, all biases and layernorms don't
    decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
      {'params': decay_params, 'weight_decay': weight_decay},
      {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f'Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters')
    print(f'Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters')
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and 'cuda' in device
    print(f'Using fused AdamW: {use_fused}')
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer

  @torch.no_grad()  
  def generate(self, prompt=None, max_new_tokens=25, topk=50):
    '''
    if prompt is None:
      # Start with a random token (not special tokens like <|endoftext|>)
      vocab_size = encoding.n_vocab  # Number of tokens in the vocabulary
      # Ensure the random token is not the <|endoftext|> token (assuming it's the last token)
      start_token = torch.randint(0, vocab_size - 1, (1, 1), dtype=torch.long, device=device)  # Random token (excluding last special token)
      prompt = start_token
    else:
      # Ensure prompt is a tensor and move to the correct device
      if isinstance(prompt, list):
          prompt = torch.tensor(prompt, dtype=torch.long, device=device)
      if len(prompt.shape) == 1:
          prompt = prompt.unsqueeze(0)  # Add batch dimension if missing
    '''
    for _ in range(max_new_tokens):
      # Crop prompt if it exceeds block_size
      if prompt.size(1) > self.config.block_size:
        prompt = prompt[:, -self.config.block_size:]

      # Forward pass
      logits, _ = self(prompt)
      logits = logits[:, -1, :]  # Take the last token's logits

      # Sample from top-k tokens
      probs = F.softmax(logits, dim=-1)
      topk_probs, topk_indices = torch.topk(probs, topk, dim=-1)
      index = torch.multinomial(topk_probs, 1)
      next_token = torch.gather(topk_indices, -1, index)

      # Append to the prompt
      prompt = torch.cat((prompt, next_token), dim=1)

    return prompt