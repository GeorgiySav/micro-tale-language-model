'''from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import pandas as pd

train_set = pd.read_csv('dataset/train.csv', encoding='utf-8')
test_set = pd.read_csv('dataset/validation.csv', encoding='utf-8')

with open('dataset/stories.txt', 'w', encoding='utf-8') as f:
  for row in train_set['text']:
    f.write(str(row)+'\n')
  for row in test_set['text']:
    f.write(str(row)+'\n')

tokenizer = Tokenizer(BPE(unk_token='<|unknown|>'))
trainer = BpeTrainer(special_tokens=["<|unknown|>", "<|im_start|>", "<|im_end|>"], vocab_size=2024, min_frequency=1)
tokenizer.pre_tokenizer = Whitespace()

tokenizer.train(['dataset/stories.txt'], trainer)
tokenizer.save('tokenizer.json')'''

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

dataset = load_dataset('csv', data_files={'train': 'dataset/train.csv'})
dataset = dataset.filter(lambda x : isinstance(x['text'], str))

for text in dataset['train']['text']:
  if not isinstance(text, str):
    print('Found none value')

print(f'<|start_story|>{dataset['train']['text'][0]}<|end_story|>')

def get_training_corpus():
  for i in range(0, len(dataset['train']), 1000):
    yield dataset['train']['text'][i:i+1000]

def filter(x):
  return [f'{text} ' for text in x] 

'''training_corpus = [
  f'{text} ' for i in tqdm(range(0, len(dataset['train']['text']), 4096)) for text in dataset['train']['text'][i:i+4096] 
]'''
print('Made training corpus')

base_tokenizer = AutoTokenizer.from_pretrained('empty_tokenizer')
tokenizer = base_tokenizer.train_new_from_iterator(filter(get_training_corpus()), 2048)

example = """<|start_story|>Once upon a time, in a big forest, there lived a rhinoceros named Roxy. Roxy loved to climb. She climbed trees, rocks, and hills. One day, Roxy found an icy hill. She had never seen anything like it before. It was shiny and cold, and she wanted to climb it. Roxy tried to climb the icy hill, but it was very slippery. She tried again and again, but she kept falling down. Roxy was sad. She wanted to climb the icy hill so much. Then, she saw a little bird named Billy. Billy saw that Roxy was sad and asked, "Why are you sad, Roxy?" Roxy told Billy about the icy hill and how she couldn't climb it. Billy said, "I have an idea! Let's find some big leaves to put under your feet. They will help you climb the icy hill." Roxy and Billy looked for big leaves and found some. Roxy put the leaves under her feet and tried to climb the icy hill again. This time, Roxy didn't slip. She climbed and climbed until she reached the top of the icy hill. Roxy was so happy! She and Billy played on the icy hill all day. From that day on, Roxy and Billy were the best of friends, and they climbed and played together all the time. And Roxy learned that with a little help from a friend, she could climb anything. <|end_story|>"""
print(example)
print()

tokens = tokenizer.tokenize(example)
print(tokens)
print(len(tokens))
print()

encoded = tokenizer(example, add_special_tokens=False)
print(encoded)
print()

decoded = tokenizer.decode(encoded['input_ids'])
print(decoded)
