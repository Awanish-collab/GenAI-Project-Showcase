import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden_size)
        # encoder_outputs: (batch, seq_len, hidden_size)
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_size)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch, seq_len, hidden_size)
        energy = energy.transpose(1, 2)  # (batch, hidden_size, seq_len)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # (batch, 1, hidden_size)
        attention = torch.bmm(v, energy).squeeze(1)  # (batch, seq_len)
        return F.softmax(attention, dim=1)  # (batch, seq_len)

class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embed_size=128, hidden_size=256, num_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_size * 2, output_vocab_size)
        self.attention = Attention(hidden_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        x = x.unsqueeze(1)  # (batch, 1)
        embedded = self.embedding(x)  # (batch, 1, embed_size)

        attn_weights = self.attention(hidden[-1], encoder_outputs)  # (batch, seq_len)
        attn_weights = attn_weights.unsqueeze(1)  # (batch, 1, seq_len)
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch, 1, hidden_size)

        rnn_input = torch.cat((embedded, context), dim=2)  # (batch, 1, embed + hidden)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))  # output: (batch, 1, hidden)

        output = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))  # (batch, vocab_size)
        return output, hidden, cell
