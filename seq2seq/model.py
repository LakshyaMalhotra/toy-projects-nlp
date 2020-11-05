import random

import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_token = 0


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, embedding_dim=embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, batch_first=True)

    def forward(self, input_enc):
        embedded = self.embed(input_enc)
        # print(f"Output shape of encoder embedding: {embedded.size()}")
        output, hidden = self.rnn(embedded)

        return output, hidden
        # output size = [batch_size, input_length,  hidden_size * n_directions]

    # def init_hidden(self, batch_size=1):
    #     return torch.zeros(1, batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_dim, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embed = nn.Embedding(output_size, embedding_dim=embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_dec, hidden):
        # input_dec size = [batch_size, output_size (size of output vocab)]
        embedded = self.embed(input_dec)
        # embedded size = [batch_size, 1, embed_dim]

        # print(f"Output shape of decoder embedding: {embedded.size()}")
        embedded = F.relu(embedded)

        output, hidden = self.rnn(embedded)
        # output size = [batch_size, 1, hidden_dim]
        # hidden_dim = [n_layers*n_directions, batch_size, hidden_size]

        output = F.log_softmax(self.linear(output[0]), dim=1)
        # output size = [1, output_size]
        return output, hidden

    # def init_hidden(self, batch_size=1):
    #     return torch.zeros(1, batch_size, self.hidden_size, device=device)


class Seq2Seq(nn.Module):
    def __init__(
        self, encoder, decoder, target_vocab_size, device=torch.device("cpu")
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab_size = target_vocab_size
        self.device = device

        assert (
            encoder.hidden_size == decoder.hidden_size
        ), "Hidden size of encoder and decoder must be equal"

    def forward(
        self, encoder_input, target_tensor, teacher_forcing_ratio=0.5,
    ):
        # tensor to store the decoder outputs
        # shape of target tensor: (batch_size, target_length)
        target_length = target_tensor.size(1)
        outputs = torch.zeros(
            1, target_length, self.target_vocab_size, device=self.device
        )

        # We do not provide any initial hidden state to encoder since PyTorch
        # by default initializes it to zeros if not provided
        encoder_output, encoder_hidden = self.encoder(encoder_input)

        # first input to the decoder is the <SOS> tokens
        decoder_input = torch.LongTensor([[SOS_token]])
        decoder_hidden = encoder_hidden

        for di in range(target_length):
            output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            # place predictions in a tensor holding predictions for each token
            outputs[0][di] = output

            # decide if we want to use teacher forcing or not
            teacher_forcing = (
                True if random.random() < teacher_forcing_ratio else False
            )

            # get the highest prediction token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as the decoder input
            # else, use the prediction
            decoder_input = (
                target_tensor[0][di].view(-1, 1)
                if teacher_forcing
                else top1.view(-1, 1)
            )

        return outputs

