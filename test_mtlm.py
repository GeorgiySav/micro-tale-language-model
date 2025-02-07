# imports
import torch
from tokenizers import Tokenizer, decoders

from model import GPT, GPTConfig
# ------------------------------------

# setup the device
device = 'cpu'
if torch.cuda.is_available():
  device = 'cuda'
print(f'Using device: {device}')
if device == 'cuda':
  print(f'Device name: {torch.cuda.get_device_name(device=device)}')
torch.set_float32_matmul_precision('high')
# -------------------------------------

# load dataset
tokenizer_file = 'tokenizer.json'
tokenizer = Tokenizer.from_file(tokenizer_file)
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(
  tokenizer_object=tokenizer,
  bos_token='<|start_story|>',
  eos_token='<|end_story|>',
)
# -------------------------------------

# load model
checkpoint = torch.load("checkpoints/mtlm-complete.pt", weights_only=True)
hp = checkpoint['hyperparameters']
block_size = hp['block_size']
vocab_size = hp['vocab_size']
num_layers = hp['num_layers']
num_heads  = hp['num_heads']
num_embed  = hp['num_embed']
dropout    = hp['dropout']
config = GPTConfig(
  block_size = block_size,
  vocab_size = vocab_size,
  num_layers = num_layers,
  num_heads  = num_heads,
  num_embed  = num_embed,
  dropout    = dropout
)
model = GPT(config=config)
model.to(device)
model.load_state_dict(checkpoint['model'])
# -----------------------------

# generate a micro tale
prompt = 'Once upon a time, '
prompt = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
response = model.generate(prompt=prompt, max_new_tokens=500, topk=50)
tokens = response[0].tolist()
decoded = tokenizer.decode(tokens, skip_special_tokens=True)
print(decoded)