# imports
import json
import math
import time

import torch

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from model import GPT, GPTConfig
from dataloader_ts import DataLoaderTS
# -------------------------------------


# setup the device
device = 'cpu'
if torch.cuda.is_available():
  device = 'cuda'
print(f'Using device: {device}')
if device == 'cuda':
  print(f'Device name: {torch.cuda.get_device_name(device=device)}')
torch.set_float32_matmul_precision('high')
# -------------------------------------


# hyperparameters
with open('hyperparameters.json') as f:
  hp = json.load(f)
  print(json.dumps(hp, indent=2))

block_size = hp['block_size']
vocab_size = hp['vocab_size']
num_layers = hp['num_layers']
num_heads  = hp['num_heads']
num_embed  = hp['num_embed']
dropout    = hp['dropout']

B = hp["mini_batch_size"]
T = block_size
total_batch_size = hp['total_batch_size']
assert total_batch_size % (B*T) == 0, 'Make sure total_batch_size is divisible by B*T'
grad_accum_steps = total_batch_size // (B*T)
print(f'Calculate gradient accumulation steps: {grad_accum_steps}')

max_lr = hp["learning_rate"]
min_lr = max_lr * 0.1
warmup_steps = hp["warmup_steps"]
max_steps = hp["max_steps"]
weight_decay = hp["weight_decay"]
grad_clipping = hp["grad_clipping"]

val_interval = hp['val_interval']
val_steps = hp['val_steps']
gen_interval = hp['val_interval']
checkpoint_interval = hp['checkpoint_interval']
# -------------------------------------


# load dataset
tokenizer_file = 'tokenizer.json'
tokenizer = Tokenizer.from_file(tokenizer_file)
wrapped_tokenizer = PreTrainedTokenizerFast(
  tokenizer_object=tokenizer,
  bos_token='<|start_story|>',
  eos_token='<|end_story|>',
  pad_token='<|end_story|>',
  unk_token='<|unk|>',
)

dataloader = DataLoaderTS(B, T, tokenizer=wrapped_tokenizer)
# -------------------------------------


# model
config = GPTConfig(
  block_size = block_size,
  vocab_size = vocab_size,
  num_layers = num_layers,
  num_heads  = num_heads,
  num_embed  = num_embed,
  dropout    = dropout
)
model = GPT(config)
model.to(device)

checkpoint = torch.load('checkpoints/mtlm-complete-0602.pt', weights_only=True)
model.load_state_dict(checkpoint['model'])
# -------------------------------------


# training
def get_lr(it):
  # 1) linear warmup for warmup_ters steps
  if it < warmup_steps:
    return max_lr * (it+1) / warmup_steps
  # 2) if it > max_steps, return min learning rate
  if it > max_steps:
    return min_lr
  # 3) in between, use cosine decay down to min learning rate
  decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
  return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device=device)

for step in range(max_steps):
  t0 = time.time()

  if step % val_interval == 0 and step != 0:
    val_loss_accum = 0.0
    model.eval()
    with torch.no_grad():
      for _ in range(val_steps):
        x, y = dataloader.next_batch(split='test')
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss = loss / val_steps
        val_loss_accum += loss.detach()
    print(f'Validation loss: {val_loss_accum.item():.4f}')

  if step % gen_interval == 0 and step != 0:
    model.eval()
    prompt = wrapped_tokenizer.encode('<|start_story|>Once upon a time, ')
    prompt = torch.tensor([prompt], dtype=torch.long, device=device)
    response = model.generate(prompt=prompt, max_new_tokens=100, topk=10)
    tokens = response[0].tolist()
    decoded = wrapped_tokenizer.decode(tokens)
    print(f'Generation: {decoded}')

  if step % checkpoint_interval == 0 and step != 0:
    checkpoint = {
      "step": step,
      "model": model.state_dict(),
      "optimizer": optimizer.state_dict(),
      "hyperparameters": hp
    }
    torch.save(checkpoint, f"checkpoints/mtlm-checkpoint-{step}.pt")

  model.train()
  optimizer.zero_grad()
  loss_accum = 0.0
  for micro_step in range(grad_accum_steps):
    x, y = dataloader.next_batch(split='train')
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
    loss = loss / grad_accum_steps
    loss_accum += loss.detach()
    loss.backward()

  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)
  # determine learning rate
  lr = get_lr(step)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr 
  optimizer.step()

  if device == 'cuda': 
    torch.cuda.synchronize()

  t1 = time.time()
  dt = t1 - t0
  tokens_per_sec = (B * T * grad_accum_steps) / dt
  print(f'Step {step:4d} | Loss: {loss_accum.item():.6f} | Lr: {lr:.4e} | Norm: {norm:.4f} | Time: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec}')
# -------------------------------------

# save model
checkpoint = {
  "step": max_steps,
  "model": model.state_dict(),
  "optimizer": optimizer.state_dict(),
  "hyperparameters": hp
}
torch.save(checkpoint, f"checkpoints/mtlm-complete.pt") 
# -------------------------------------