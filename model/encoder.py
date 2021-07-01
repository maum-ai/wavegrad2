import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = list()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(0.5),
            ))
        self.cnn = nn.Sequential(*self.cnn)

        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        x = self.cnn(x)  # [B, chn, T]
        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)

        x = F.dropout(x, 0.5, self.training)

        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x
        


