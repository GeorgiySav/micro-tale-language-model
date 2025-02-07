import torch
import pandas as pd

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

class DataLoaderTS:
  def __init__(self, B, T, tokenizer : PreTrainedTokenizerFast):
    dataset = load_dataset('csv', data_files={'train': 'dataset/train.csv', 'validation': 'dataset/validation.csv'})
    dataset = dataset.filter(lambda x : type(x['text']) == str)

    tokenized = dataset.map(lambda x : {
      'input_ids' :  tokenizer.batch_encode_plus([f'<|start_story|>{elem}<|end_story|>' for elem in x['text']], padding='max_length', max_length=T+1, truncation=True)['input_ids']
    }, batched=True)
    tokenized.set_format(type='torch')

    train_ids = tokenized['train'].remove_columns(['text'])
    train_ids = train_ids.shuffle().select(range(30000))

    val_ids = tokenized['validation'].remove_columns(['text'])
    val_ids = val_ids.shuffle().select(range(3000))

    print(f'Loaded {len(train_ids)} training stories')
    print(f'Loaded {len(val_ids)} validation stories')

    self.train_dataloader = DataLoader(train_ids, batch_size=B, shuffle=True)
    self.val_dataloader = DataLoader(val_ids, batch_size=B, shuffle=True)

  def next_batch(self, split):
    dl = self.train_dataloader if split == 'train' else self.val_dataloader
    batch = next(iter(dl))['input_ids']
    x = batch[:, :-1]
    y = batch[:, 1:]
    return x, y