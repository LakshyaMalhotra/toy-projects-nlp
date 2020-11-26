import random

import torch
import torch.nn as nn

## Seq2seq model:
## Encoder: simple embedding layer followed by a GRU layer
class Encoder(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(
            num_embeddings=input_size, embedding_dim=embed_dim
        )
        self.rnn = nn.GRU(embed_dim, hidden_size)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input):
        # input shape: [seq_len, batch_size]
        embedded = self.dropout(self.embed(input))
        # embedded shape: [seq_len, batch_size, embed_dim]

        output, hidden = self.rnn(embedded)
        # output shape: [seq_len, batch_size, hidden_size]
        # hidden shape: [num_layers*num_directions, batch_size, hidden_size]

        # For encoder, we are not interested in the output so skipping returning it
        return hidden


## Decoder: this is where magic happens
class Decoder(nn.Module):
    def __init__(self, output_size, embed_dim, hidden_size):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(
            num_embeddings=output_size, embedding_dim=embed_dim
        )
        self.rnn = nn.GRU((embed_dim + hidden_size), hidden_size)
        self.linear = nn.Linear((embed_dim + 2 * hidden_size), output_size)
        self.dropout = nn.Dropout(p=0.3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, context_vector):
        # For the decoder: seq_len, num_layers, num_directions == 1
        # input shape: [batch_size]

        input = input.unsqueeze(0)
        # input shape: [seq_len, batch_size]
        # hidden shape: [num_layers*num_directions, batch_size, hidden_size]
        # context_vector shape: hidden shape

        embedded = self.dropout(self.embed(input))
        # embedded shape: [seq_len, batch_size, embed_dim]

        # input to RNN: it takes both embedding layer output and context vector
        # at each time step
        rnn_in = torch.cat((embedded, context_vector), -1)
        rnn_out, hidden = self.rnn(rnn_in)
        # output shape: [seq_len, batch_size, hidden_size]
        # hidden shape: [num_layers*num_directions, batch_size, hidden_size]

        linear_in = torch.cat((rnn_out, embedded, context_vector), -1)
        linear_in = linear_in.squeeze(0)
        # linear_in shape: [batch_size, (embed_dim + 2*hidden_size)]

        linear_out = self.linear(linear_in)
        # linear_out shape: [batch_size, output_size]

        out = self.softmax(linear_out)
        # out shape: [batch_size, output_size]
        return out, hidden


# Encapsulating both encoder and decoder
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device=torch.device("cpu")):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_tensor, target_tensor, teacher_forcing_ratio=0.5):
        # input_tensor shape: [seq_len, batch_size]
        # target_tensor shape: [seq_len, batch_size]
        batch_size = target_tensor.size(-1)
        target_length = target_tensor.size(0)

        output_size = self.decoder.output_size
        # size of the target vocabulary

        # container to hold the predictions for each target
        outputs = torch.zeros(target_length, batch_size, output_size).to(
            self.device
        )

        # we don't provide any hidden state to the encoder since pytorch
        # by default initializes it to zeros if not provided
        encoder_hidden = self.encoder(input_tensor)

        # context vector is just the encoder hidden. In this decoder architecture,
        # context vector is fed into RNN at every time step.
        # For the first time step though, the hidden state for the decoder is
        # also the context vector
        context_vector = encoder_hidden
        decoder_hidden = encoder_hidden

        # decoder input for the first time step is just `<sos>` token
        decoder_input = target_tensor[0, :]

        # iterate through the target tensor
        for ti in range(1, target_length):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, context_vector
            )

            # decoder_output shape: [batch_size, output_size]
            # add the current time step output to outputs
            outputs[ti] = decoder_output

            # get the top prediction
            top1 = decoder_output.argmax(1)
            # top1 shape: [batch_size]

            # decide if we are using teacher forcing for the next time step
            teacher_forcing = (
                True if random.random() < teacher_forcing_ratio else False
            )

            # if teacher forcing, then use the target token at the next time
            # step to be the decoder input, else use output
            decoder_input = target_tensor[ti] if teacher_forcing else top1

        return outputs