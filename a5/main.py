import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torchtext
import torchtext.data

import models

def plot_history(train_loss, train_acc, val_loss, val_acc):
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.plot(train_loss, label='train')
    if val_loss is not None:
        plt.plot(val_loss, label='validation')
        plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.plot(train_acc, label='train')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    if val_acc is not None:
        plt.plot(val_acc, label='validation')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()

def load_dataset(overfit, batch_size, device):
    field_text = torchtext.data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
    field_label = torchtext.data.Field(sequential=False, use_vocab=False)
    fields = [('text', field_text), ('label', field_label)]

    if not overfit:
        train_data, val_data, test_data = torchtext.data.TabularDataset.splits(
                path='./data', train='train.tsv', validation='validation.tsv', test='test.tsv',
                format='TSV', skip_header=True, fields=fields,
        )
        field_text.build_vocab(train_data, val_data, test_data)
        iters = torchtext.data.BucketIterator.splits(
                (train_data, val_data, test_data),
                batch_sizes=(batch_size, batch_size, batch_size),
                sort_key=lambda x: len(x.text),
                device=device,
                sort_within_batch=True,
                repeat=False,
        )

    else:
        train_data = torchtext.data.TabularDataset('./data/overfit.tsv', format='tsv',
                skip_header=True, fields=fields)
        field_text.build_vocab(train_data)
        iters = (
            torchtext.data.BucketIterator(
                train_data, batch_size, sort_key=lambda x: len(x.text),
                device=device, sort_within_batch=True, repeat=False
            ),
            None,
            None
        )

    field_text.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    return field_text.vocab, iters

def calculate_accuracy(pred, labels):
    return float(((pred >= 0) == labels).sum().item()) / labels.size(0)

def get_model_name(overfit, model_name, lr, batch_size, epochs):
    if overfit:
        return f'overfit_{model_name}_{lr}lr_{batch_size}batch_{epochs}epochs'
    else:
        return f'{model_name}_{lr}lr_{batch_size}batch_{epochs}epochs'

def validate_model(model, val_iter):
    avg_loss = 0
    avg_acc = 0
    steps = 0

    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    for batch in val_iter:
        inputs, lengths = batch.text
        labels = batch.label.float()

        pred = model(inputs, lengths)
        avg_loss += float(criterion(pred, labels))
        avg_acc += calculate_accuracy(pred, labels)

        steps += 1

    return avg_loss / steps, avg_acc / steps

def test_model(model, test_iter):
    avg_acc = 0
    steps = 0

    model.eval()
    for batch in test_iter:
        inputs, lengths = batch.text
        labels = batch.label.float()
        
        pred = model(inputs, lengths)
        avg_acc += calculate_accuracy(pred, labels)

        steps += 1

    return avg_acc / steps

def train_model(model, iters, lr, epochs):
    train_iter, val_iter, _ = iters

    train_loss = []
    train_acc = []
    if val_iter:
        val_loss = []
        val_acc = []
    else:
        val_loss = None
        val_acc = None

    optim = torch.optim.Adam(model.parameters(), lr)
    criterion = nn.BCEWithLogitsLoss()
    for i in range(1, epochs + 1):
        steps = 0
        avg_loss = 0
        avg_acc = 0

        model.train()
        for batch in train_iter:
            inputs, lengths = batch.text
            labels = batch.label.float()
            optim.zero_grad()
            pred = model(inputs, lengths)
            loss = criterion(pred, labels)
            loss.backward()
            optim.step()

            avg_loss += float(loss)
            avg_acc += calculate_accuracy(pred, labels)
            steps += 1

        avg_loss /= steps
        avg_acc /= steps

        train_loss.append(avg_loss)
        train_acc.append(avg_acc)

        if val_iter:
            with torch.no_grad():
                loss, acc = validate_model(model, val_iter)

            val_loss.append(loss)
            val_acc.append(acc)
            print("{}\ttrain loss {:.4f}\tacc {:.4f}\tval loss {:.4f}\t acc {:.4f}".format(
                i, avg_loss, avg_acc, loss, acc
            ))
        else:
            print("{}\ttrain loss {:.4f}\tacc {:.4f}".format(
                i, avg_loss, avg_acc
            ))

    return train_loss, train_acc, val_loss, val_acc

def main(model_name, lr, epochs, batch_size, overfit, test, device):
    print('Training on', device)

    vocab, iters = load_dataset(overfit, batch_size, device)
    print('Shape of vocab:', vocab.vectors.shape)

    model_cls = models.CLASS_DICT[model_name]
    model = model_cls(vocab).to(device)

    start = time.time()
    history = train_model(model, iters, lr, epochs)
    end = time.time()

    model_file = get_model_name(overfit, model_name, lr, batch_size, epochs)

    print('Max validation accuracy:', np.max(history[3]))
    print('Trained in:', end - start)
    
    if test:
        accuracy = test_model(model, iters[2])
        print('Testing accuracy:', accuracy)

    plot_history(*history)
    plt.savefig(f'figures/{model_file}.svg')

    torch.save(model.cpu(), f'models/{model_file}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--model', dest='model_name', type=str, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available else 'cpu'
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    plt.rcParams["figure.figsize"] = (10,5)

    main(args.model_name, args.lr, args.epochs, args.batch_size, args.overfit, args.test, device)

