import tiktoken
import torch

encoding = tiktoken.get_encoding('gpt2')

class DataLoader:
  def __init__(self, B, T, process_rank, num_processes, test_split=0.1):
    self.B = B
    self.T = T
    self.process_rank = process_rank
    self.num_processes = num_processes

    # at initialisation, load data and store them in memory
    with open('input.txt', 'r') as f:
      text = f.read()
    tokens = encoding.encode(text)
    tokens = torch.tensor(tokens)
    print(f'Loaded {len(tokens)} tokens')
    print(f'1 epoch = {len(tokens) // (B*T)} batches')
    split_index = int(len(tokens) * test_split)
    self.train_tokens = tokens[split_index:]
    self.test_tokens = tokens[:split_index]

    # state
    self.current_position = self.B * self.T * self.process_rank
  
  def size(self, split):
    if split == 'train':
      return len(self.train_tokens)
    return len(self.test_tokens)

  def next_batch(self):
    B, T = self.B, self.T
    buffer = self.train_tokens[self.current_position: self.current_position + (B*T) + 1]
    x = (buffer[:-1]).view(B, T)
    y = (buffer[1:]).view(B, T)

    self.current_position += B*T*self.num_processes

    if self.current_position + (B*T*self.num_processes + 1) > len(self.train_tokens):
      self.current_position = self.B * self.T * self.process_rank
    
    return x, y
  
  def val_batch(self):
    B, T = self.B, self.T
    # get random position in the test_tokens
    index = torch.randint(len(self.test_tokens)-B*T-1, (1,)).item()
    
    buffer = self.test_tokens[index:index + (B*T) + 1]
    x = (buffer[:-1]).view(B, T)
    y = (buffer[1:]).view(B, T)

    return x, y