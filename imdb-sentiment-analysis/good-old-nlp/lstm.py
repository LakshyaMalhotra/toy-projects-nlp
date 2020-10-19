# lstm.py
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, embedding_matrix):
        super(LSTM, self).__init__()

        # getting the number of words and embedding dimension from embedding matrix
        num_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]

        # input embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=num_words, embedding_dim=embed_dim
        )

        # embedding matrix is used as weights of the embedding layer
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )

        # we don't want to train the pretrained embeddings
        self.embedding.weight.requires_grad = False

        # defining a simple LSTM with 128 hidden units
        self.lstm = nn.LSTM(
            embed_dim, 128, bidirectional=True, batch_first=True
        )

        # output is the linear layer
        self.out = nn.Linear(512, 1)

    def forward(self, x):
        # pass data through embedding layer
        x = self.embedding(x)

        # move embedding output to lstm
        x, _ = self.lstm(x)

        # apply mean and max
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)

        # concatenate the max and average pooling
        # out size is 512
        # 2 x 128 each for max and avg pooling (2 for bidirectional lstm)
        out = torch.cat((avg_pool, max_pool), 1)

        # pass through the linear layer
        out = self.out(out)

        return out