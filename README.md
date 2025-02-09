# Micro Tale Language Model

A small language model based on the GPT-2 architecture and trained on the Tiny Stories Dataset.
The tokenizer was also trained on the dataset in order to reduce it to a size of 2048.

## Dependencies

Does require the tiny stories dataset in csv format, which can be [downloaded here](https://www.kaggle.com/datasets/thedevastator/tinystories-narrative-classification).

Requires:
```bash
pip install torch datasets transformers tokenizers
```
## Training
All code for training is provided. Edit the `hyperparameters.json` file to change the training and model parameters. Also, in the `dataloader_ts.py` file, I only selected a small subset of dataset.

I trained a 3.815 million parameters model, and managed to achieve `0.7083 validation loss`.

Training was done on a laptop gtx 1650 with 4GB of VRAM and took 8 hours to complete.

## Sample
```
Once upon a time, there was a little boy named Tim. Tim was eager to play at the beach. He liked to run and jump and play with his toy car. One day, Tim saw a big wave in the water. He wanted to play in it, but he was scared.

His dad came to him and said, "Don't worry, Tim. The water might hurts." Tim tried the wave to, but he was too scared. He just wanted to play.

One day, Tim saw his dad in the sand. He walked over to her and said, "Hi, Daddy! What are you doing?" His dad said, "I'm scared. Do you want to play too?" Tim nodded his head and said, "Yes, I want to be scared".

Tim and his dad ran to the sand. They played together in the sand. Tim laughed and said, "See, I saw the sight!" His dad smiled and said, "You're okay, Tim. You're safe." The dad smiled and said, "See, you're very brave too." 
```

Does produce text that is somewhat coherent, though nothing special. I can improve the model further by training it on the rest of the dataset, however, due to time constraints, that's not in the picture.

## License

[MIT](https://choosealicense.com/licenses/mit/)