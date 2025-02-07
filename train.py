import math
import os
import time
from model import GPT, GPTConfig
from dataloader import DataLoader, encoding

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# hyperparameters
gpt_settings = GPTConfig(
  block_size = 256,
  #vocab_size = 50304,
  num_layers = 2,
  num_heads = 4,
  num_embed = 256,
  dropout = 0.1
)
torch.set_float32_matmul_precision('high')

B = 4
T = gpt_settings.block_size

max_lr = 1e-3
min_lr = max_lr * 0.1
warmup_steps = 100
max_steps = 2000

weight_decay = 0.1

# device
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
  # use of ddp atm demands CUDA, we set the device appropriately according to rank
  assert torch.cuda.is_available(), 'We need CUDA for DDP'
  init_process_group(backend='nccl')
  ddp_rank = int(os.environ['RANK'])
  ddp_local_rank = int(os.environ['LOCAL_RANK'])
  ddp_world_size = int(os.environ['WORLD_SIZE'])
  device = f'cuda:{ddp_local_rank}'
  torch.cuda.set_device(device)
  master_process = ddp_rank == 0 # for logging
else:
  # no ddp
  ddp_rank = 0
  ddp_local_rank = 0
  ddp_world_size = 1
  master_process = True

  device = 'cpu'
  if torch.cuda.is_available():
    device = 'cuda'
  print(f'Using device: {device}')


# load dataset
data_loader = DataLoader(B, T, process_rank=ddp_rank, num_processes=ddp_world_size)

total_batch_size = data_loader.size(split='train') // (B*T)
total_batch_size *= B*T
assert total_batch_size % (B*T*ddp_world_size) == 0, 'Make sure total_batch_size is divisible by B*T*ddp_world_size'
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
  print(f'Total desired batch size: {total_batch_size}')
  print(f'Calculated gradient accumulation steps: {grad_accum_steps}')



# create model
model = GPT(gpt_settings)
model.to(device)
#model = torch.compile(model)
if ddp:
  model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model


# determine learning rate throughout the training process
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


# train model
optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device=device)
for step in range(max_steps):
  t0 = time.time()

  # once in while, evaluate our validation loss
  if step % 10 == 0:
    model.eval()
    with torch.no_grad():
      val_loss_accum = 0.0
      val_loss_steps = 10
      for _ in range(val_loss_steps):
        x, y = data_loader.next_batch()
        x, y = x.to(device), y.to(device)
        #with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
        loss = loss / val_loss_steps
        val_loss_accum += loss.detach()
    if ddp:
      dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
      print(f'Validation loss: {val_loss_accum.item():.4f}')

  if step % 50 == 0 and step != 0:
    model.eval()
    response = model.generate(max_new_tokens=32, encoding=encoding, device=device)
    tokens = response[0].tolist()
    decoded = encoding.decode(tokens)
    print(f'Rank {ddp_rank}: {decoded}')

  model.train()
  optimizer.zero_grad()
  loss_accum = 0.0
  for micro_step in range(grad_accum_steps):
    x, y = data_loader.next_batch()
    x, y = x.to(device), y.to(device)
    #with torch.autocast(device_type=device, dtype=torch.bfloat16):
    logits, loss = model(x, y)
    loss = loss / grad_accum_steps
    loss_accum += loss.detach()
    if ddp:
      model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
    loss.backward()

  if ddp:
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  # determine learning rate
  lr = get_lr(step)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr 
  optimizer.step()

  torch.cuda.synchronize()
  t1 = time.time()
  dt = t1 - t0
  tokens_per_sec = (data_loader.B * data_loader.T * grad_accum_steps * ddp_world_size) / dt
  if master_process:
    print(f'Step {step:4d} | Loss: {loss_accum.item():.6f} | Lr: {lr:.4e} | Norm: {norm:.4f} | Time: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec}')


# destroy devices
if ddp:
  destroy_process_group()