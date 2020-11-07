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

    def forward(self, input_enc, input_len):
        embedded = self.embed(input_enc)
        # print(f"Output shape of encoder embedding: {embedded.size()}")
        embed_packed = nn.utils.rnn.pack_padded_sequence(
            embedded, input_len, batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.rnn(embed_packed)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)

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
        # embedded size = [batch_size, seq_len, embed_dim]

        # print(f"Output shape of decoder embedding: {embedded.size()}")
        embedded = F.relu(embedded)

        output, hidden = self.rnn(embedded, hidden)
        # print(output.size())
        # output size = [batch_size, seq_len, n_directions*hidden_dim]
        # hidden_dim = [n_layers*n_directions, batch_size, hidden_size];
        # batch size is in `dim=1` of the hidden size even after setting `batch_first`=True

        output = F.log_softmax(self.linear(output.squeeze(1)), dim=1)
        # output size = [batch_size, output_size]
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
        self,
        encoder_input,
        target_tensor,
        input_len,
        teacher_forcing_ratio=0.5,
    ):
        # tensor to store the decoder outputs
        # shape of target tensor: (batch_size, target_length)
        target_length = target_tensor.size(1)
        batch_size = target_tensor.size(0)
        outputs = torch.zeros(
            batch_size, target_length, self.target_vocab_size
        ).to(self.device)

        # We do not provide any initial hidden state to encoder since PyTorch
        # by default initializes it to zeros if not provided
        _, encoder_hidden = self.encoder(encoder_input, input_len)

        # first input to the decoder is the <SOS> tokens
        sos_tensor = torch.zeros(batch_size, 1, dtype=torch.long)
        # decoder_input = torch.LongTensor([[SOS_token]]).to(self.device)
        decoder_input = torch.LongTensor(sos_tensor).to(self.device)
        decoder_hidden = encoder_hidden

        for di in range(target_length):
            output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            # place predictions in a tensor holding predictions for each token
            outputs[:, di, :] = output

            # decide if we want to use teacher forcing or not
            teacher_forcing = (
                True if random.random() < teacher_forcing_ratio else False
            )

            # get the highest prediction token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as the decoder input
            # else, use the prediction
            decoder_input = (
                target_tensor[:, di].view(-1, 1)
                if teacher_forcing
                else top1.view(-1, 1)
            )

        return outputs
