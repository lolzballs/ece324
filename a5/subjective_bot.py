"""
    6. Testing on Your Own Sentence.

    You will write a python script that prompts the user for a sentence input on the command line, and prints the
    classification from each of the three models, as well as the probability that this sentence is subjective.

    An example console output:

        Enter a sentence
        What once seemed creepy now just seems campy

        Model baseline: subjective (0.964)
        Model rnn: subjective (0.999)
        Model cnn: subjective (1.000)

        Enter a sentence
"""

import collections
import readline
import spacy
import torch
import torchtext

import models

def load_vocab():
    field_text = torchtext.data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
    field_label = torchtext.data.Field(sequential=False, use_vocab=False)
    fields = [('text', field_text), ('label', field_label)]
    data = torchtext.data.TabularDataset.splits(
            path='./data', train='train.tsv', validation='validation.tsv', test='test.tsv',
            format='TSV', skip_header=True, fields=fields,
    )

    field_text.build_vocab(*data)
    field_text.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    return field_text.vocab

def tokenizer(text):
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en(text)]

def classify(model, token_ints):
    token_tensor = torch.LongTensor(token_ints).view(-1, 1)
    lengths = torch.Tensor([len(token_ints)])
    pred = torch.sigmoid(model(token_tensor, lengths))
    return 'subjective' if pred > 0.5 else 'objective', float(pred)

baseline = torch.load('./model_baseline.pt')
cnn = torch.load('./model_cnn.pt')
rnn = torch.load('./model_rnn.pt')

vocab = load_vocab()

while True:
    sentence = input('Enter a sentence\n')
    tokens = tokenizer(sentence.lower())
    token_ints = [vocab.stoi[tok] for tok in tokens]

    cat, prob = classify(baseline, token_ints)
    print(f'Model baseline: {cat} ({prob:.3f})')
    cat, prob = classify(cnn, token_ints)
    print(f'Model cnn: {cat} ({prob:.3f})')
    cat, prob = classify(rnn, token_ints)
    print(f'Model rnn: {cat} ({prob:.3f})')

