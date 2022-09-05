import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        embed_dim: int,
        hidden_size: int,
        output_size: int,
    ):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x size: [batch_size, seq_len]
        embed_out = self.dropout(self.embed(x))
        # embed_out size: [batch_size, seq_len, embed_dim]

        out, _ = self.rnn(embed_out)
        # out size: [batch_size, seq_len, hidden_size]

        out = self.softmax(self.linear(out.squeeze(0)))
        # out size: [batch_size, output_size]

        # just taking the last time step output
        out = out[-1, :].view(1, -1)
        # out size: [1, output_size]

        return out

