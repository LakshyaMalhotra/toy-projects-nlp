# Library imports
import time
import glob
import os
import typing
import random

import string
import unicodedata

import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


all_letters = string.ascii_letters + "'"
n_letters = len(all_letters) + 1  # for EOS
char2idx = {char: idx for idx, char in enumerate(all_letters)}
idx2char = {idx: char for char, idx in char2idx.items()}
idx2char[n_letters - 1] = "<EOS>"

# Turn a unicode string to plain ASCII
# Strip off the accents etc.
def unicode_to_ascii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in all_letters
    )


# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Find all the files
def find_files(path):
    return glob.glob(path)


# Read the files and split into lines
def read_lines(filename):
    lines = open(filename, encoding="utf-8").read().strip().split("\n")
    return [unicode_to_ascii(line) for line in lines]


names = []
for filename in find_files("data/names/*.txt"):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = read_lines(filename)
    names += lines

total_names = len(names)
print(f"Total number of names: {total_names}")

InputTensors = typing.Tuple[torch.LongTensor, torch.LongTensor]


def tensorize(word: str) -> InputTensors:
    input_tensor = torch.LongTensor([char2idx[c] for c in word])
    eos = torch.LongTensor([n_letters - 1])
    target_tensor = torch.cat((input_tensor[1:], eos))
    return input_tensor, target_tensor


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        name_embed_dim: int,
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.name_embed = nn.Embedding(input_size, name_embed_dim)
        self.lstm = nn.LSTM(
            name_embed_dim,
            hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=0.5,
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(p=0.25)

    def forward(self, input, hidden):
        name_embed_out = self.name_embed(input)
        out, hidden = self.lstm(name_embed_out, hidden)
        out = F.log_softmax(self.drop(self.linear(out)), dim=2)

        return out, hidden

    def init_hidden(
        self,
        num_layers=2,
        num_directions=1,
        batch_size=1,
        device=torch.device("cpu"),
    ):
        if device == torch.device("cuda"):
            hidden = (
                torch.zeros(
                    num_directions * num_layers, batch_size, self.hidden_size
                ).cuda(),
                torch.zeros(
                    num_directions * num_layers, batch_size, self.hidden_size
                ).cuda(),
            )
        else:
            hidden = (
                torch.zeros(
                    num_directions * num_layers, batch_size, self.hidden_size
                ),
                torch.zeros(
                    num_directions * num_layers, batch_size, self.hidden_size
                ),
            )
        return hidden


# rnn = LSTM(
#     input_size=input_size,
#     hidden_size=hidden_size,
#     output_size=output_size,
#     name_embed_dim=name_embeddings,
# )
# device = (
#     torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# )
# rnn.to(device)
# print(rnn)
# rnn.load_state_dict(torch.load("models/names.pt"))


def sample(
    seed: str,
    max_len: int = 5,
    break_on_eos: bool = True,
    eval_mode: bool = False,
) -> str:
    # optionally set evaluation mode to disable dropout
    # if eval_mode:
    #     rnn.eval()

    input_size = n_letters
    output_size = n_letters
    hidden_size = 128
    name_embeddings = 8

    rnn = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        name_embed_dim=name_embeddings,
    )

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    rnn.to(device)
    rnn.load_state_dict(torch.load("models/names.pt"))

    # disable gradient computation
    with torch.no_grad():
        input_tensor, _ = tensorize(seed)
        hidden = rnn.init_hidden()

        # add the length-1 batch dimension to match output from Dataset
        input_tensor = input_tensor.unsqueeze(0)

        output_name = seed
        for _ in range(max_len):
            out, hidden = rnn(input_tensor, hidden)
            _, topi = out[:, -1, :].topk(
                1
            )  # grab top prediction for last character
            next_char = idx2char[int(topi.squeeze())]

            if break_on_eos and (next_char == "<EOS>"):
                break
            output_name += next_char
            input_tensor = topi

    # ensure training mode is (re-)enabled
    rnn.train()
    return output_name


letter = "M"
print(f"{letter} --> {sample(letter)}")
