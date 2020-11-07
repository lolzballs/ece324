import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self, vocab):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(vocab.vectors.size(1), 1)

    def forward(self, x, lengths=None):
        #x = [sentence length, batch size]
        embedded = self.embedding(x)

        average = embedded.mean(0) # [sentence length, batch size, embedding_dim]
        output = self.fc(average).squeeze(1)

        return output

class CNN(nn.Module):
    def __init__(self, vocab):
        super(CNN, self).__init__()
        self.embedding_dim = vocab.vectors.size(1)
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)

        self.conv_2 = nn.Conv2d(1, 50, (2, self.embedding_dim), padding=(1, 0))
        self.conv_4 = nn.Conv2d(1, 50, (4, self.embedding_dim), padding=(3, 0))
        self.fc = nn.Linear(50 * 2, 1)

    def forward(self, x, lengths):
        embeddings = self.embedding(x).transpose(0, 1).unsqueeze(1) # [L, B, E] -> [B, 1, L, E]

        filter_2 = F.relu(self.conv_2(embeddings)) # [B, 50, L, 1]
        filter_2 = F.max_pool1d(filter_2.squeeze(3), filter_2.size(2))
        filter_4 = F.relu(self.conv_4(embeddings)) # [B, 50, L, 1]
        filter_4 = F.max_pool1d(filter_4.squeeze(3), filter_4.size(2))

        x = torch.cat((filter_2, filter_4), dim=1)
        x = x.view(-1, 50 * 2)
        x = self.fc(x).squeeze(1)

        return x

class RNN(nn.Module):
    def __init__(self, vocab):
        super(RNN, self).__init__()
        self.embedding_dim = vocab.vectors.size(1)
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)

        self.gru = nn.GRU(self.embedding_dim, 100)
        self.fc = nn.Linear(100, 1)

    def forward(self, x, lengths):
        embeddings = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths)

        _, hidden = self.gru(packed)
        hidden = hidden.squeeze(0)
        x = self.fc(hidden).squeeze(1)

        return x

class RNNUnpadded(nn.Module):
    def __init__(self, vocab):
        super(RNNUnpadded, self).__init__()
        self.embedding_dim = vocab.vectors.size(1)
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)

        self.gru = nn.GRU(self.embedding_dim, 100)
        self.fc = nn.Linear(100, 1)

    def forward(self, x, lengths):
        embeddings = self.embedding(x)

        _, hidden = self.gru(embeddings)
        hidden = hidden.squeeze(0)
        x = self.fc(hidden).squeeze(1)

        return x


CLASS_DICT = {
    'baseline': Baseline,
    'cnn': CNN,
    'rnn': RNN,
    'rnn_unpadded': RNNUnpadded,
}

