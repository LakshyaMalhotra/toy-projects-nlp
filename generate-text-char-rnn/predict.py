# Library imports
import glob
import os
import typing

import string
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F

# Building the vocabulary and look up tables for characters to index and vice-versa
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


# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

for filename in find_files("data/names/*.txt"):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines
n_categories = len(all_categories)
print(f"Total categories: {n_categories}")
print(f"Categories: {all_categories}")

# Define the typing format
InputTensors = typing.Tuple[
    torch.LongTensor, torch.LongTensor, torch.LongTensor
]

# Get the category, input and target tensors for a given category and input name
def tensorize(category: str, word: str) -> InputTensors:
    cat_tensor = torch.LongTensor([all_categories.index(category)])
    input_tensor = torch.LongTensor([char2idx[c] for c in word])
    eos = torch.LongTensor([n_letters - 1])
    target_tensor = torch.cat((input_tensor[1:], eos))
    return cat_tensor, input_tensor, target_tensor


# Create Model class
class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        name_embed_dim: int,
        cat_embed_dim: int,
        n_categories: int,
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        # embedding layer for names
        self.name_embed = nn.Embedding(input_size, name_embed_dim)
        # embedding layer for categories
        self.cat_embed = nn.Embedding(n_categories, cat_embed_dim)
        # input size to RNN is the sum of embedding dim for both name and category
        self.lstm = nn.LSTM(
            (name_embed_dim + cat_embed_dim),
            hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=0.5,
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(p=0.25)

    def forward(self, input, category, hidden):
        name_embed_out = self.name_embed(input)
        cat_embed_out = self.cat_embed(category)
        # `name_embed_out` shape is `(batch_size, seq_len, name_embed_dim)` where
        # `seq_len` is the length of the name. `cat_embed_out` shape is
        # `(batch_size, 1, cat_embed_dim)`. In order to concat them together, they
        # need to have same dimensions in all but the concatenating dimension.
        # We need to repeat the `cat_embed_out` in dim=1 so that it's dimension
        # becomes `(batch_size, seq_len, cat_embed_dim)`.
        cat_embed_out = cat_embed_out.repeat(1, name_embed_out.size(1), 1)
        # concatenating the embeddings output in the last dimension
        combined = torch.cat((name_embed_out, cat_embed_out), -1)
        out, hidden = self.lstm(combined, hidden)
        # applying log softmax in the last layer containing output size
        out = F.log_softmax(self.drop(self.linear(out)), dim=2)

        return out, hidden

    # initializing the hidden layer
    def init_hidden(
        self,
        num_layers=1,
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


# Sampling the names from the trained model
def sample(
    seed: str,
    category: str,
    max_len: int = 10,
    break_on_eos: bool = True,
    eval_mode: bool = False,
) -> str:
    # define the input parameters and the model
    input_size = n_letters
    output_size = n_letters
    hidden_size = 128
    name_embeddings = 100
    cat_embeddings = 50

    rnn = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        name_embed_dim=name_embeddings,
        cat_embed_dim=cat_embeddings,
        n_categories=n_categories,
    )

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    rnn.to(device)

    # load the trained weights from the checkpoint
    rnn.load_state_dict(torch.load("models/names.pt", map_location=device))

    # optionally set evaluation mode to disable dropout
    if eval_mode:
        rnn.eval()

    # disable gradient computation
    with torch.no_grad():
        cat, input_tensor, _ = tensorize(category, seed)
        hidden = rnn.init_hidden(num_layers=2)

        # add the length-1 batch dimension to match output from Dataset
        input_tensor = input_tensor.unsqueeze(0)

        output_name = seed
        for _ in range(max_len):
            out, hidden = rnn(input_tensor, cat, hidden)
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


letter = "Sh"
print(f"{letter} --> {sample(letter, category='Japanese')}")
