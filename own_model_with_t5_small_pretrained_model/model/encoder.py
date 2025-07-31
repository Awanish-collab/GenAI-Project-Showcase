import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embed_size=128, hidden_size=256, num_layers=1, dropout=0.2):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq_len, embed_size)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell  # <--- return outputs too!

