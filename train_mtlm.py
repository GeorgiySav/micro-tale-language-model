# imports
import inspect
import json

import torch

from tokenizers import Tokenizer
from tqdm import tqdm
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
#torch.set_float32_matmul_precision('high')
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
epochs = hp["epochs"]
lr = hp["learning_rate"]

log_interval = hp['log_interval']
val_steps = hp['val_steps']
gen_interval = hp['gen_interval']
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

#checkpoint = torch.load('checkpoints/mtlm-complete.pt', weights_only=True)
#model.load_state_dict(checkpoint['model'])
# -------------------------------------


# training
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and 'cuda' in device
print(f'Using fused AdamW: {use_fused}')
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)

num_training_steps = epochs * len(dataloader.train_dataloader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=num_training_steps)

progress_bar = tqdm(range(num_training_steps), desc='Training')

for epoch in range(epochs):
  model.train()
  for batch in dataloader.train_dataloader:
    x = batch['input_ids'][:, :-1]
    y = batch['input_ids'][:, 1:]

    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
  
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    optimizer.step()
    scheduler.step()
    progress_bar.update(1)

    if progress_bar.n % log_interval == 0:
      print()
      print(f'Train loss: {loss.item():.6f} | lr: {scheduler.get_last_lr()[0]:.4e}')
    if progress_bar.n % gen_interval == 0:
      prompt = wrapped_tokenizer.encode('<|start_story|>Once upon a time, ')
      prompt = torch.tensor([prompt], dtype=torch.long, device=device)
      response = model.generate(prompt=prompt, max_new_tokens=100, topk=10)
      tokens = response[0].tolist()
      decoded = wrapped_tokenizer.decode(tokens)
      print()
      print(f'Generation: {decoded}')

  with torch.no_grad():
    model.eval()
    losses = torch.zeros(len(dataloader.val_dataloader), device=device)
    for i, batch in enumerate(dataloader.val_dataloader):
      x = batch['input_ids'][:, :-1]
      y = batch['input_ids'][:, 1:]
      x, y = x.to(device), y.to(device)
      logits, loss = model(x, y)
      losses[i] = loss.item()
    print()
    print(f'Validation loss: {loss.mean():.4f}')

  checkpoint = {
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "hyperparameters": hp
  }
  torch.save(checkpoint, f"checkpoints/mtlm-checkpoint-{epoch}-epoch.pt")
# -------------------------------------

# save model
checkpoint = {
  "epoch": epochs,
  "model": model.state_dict(),
  "optimizer": optimizer.state_dict(),
  "hyperparameters": hp
}
torch.save(checkpoint, f"checkpoints/mtlm-complete.pt") 
# -------------------------------------