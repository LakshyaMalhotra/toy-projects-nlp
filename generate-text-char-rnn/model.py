# Library imports
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


# Find all the files
def find_files(path):
    return glob.glob(path)


# Read the files and split into lines
def read_lines(filename):
    lines = open(filename, encoding="utf-8").read().strip().split("\n")
    return [unicode_to_ascii(line) for line in lines]


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


# Create PyTorch dataset
class NameDataset(torch.utils.data.Dataset):
    def __init__(self, category_lines):
        self.categories = list(category_lines.keys())
        self.names = []
        # iterating through all the categories and names
        # self.names is a list of tuples: (category_name, name)
        for cat in self.categories:
            names = category_lines[cat]
            for name in names:
                self.names.append((cat, name))

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, item: int) -> InputTensors:
        return tensorize(*self.names[item])


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
    # optionally set evaluation mode to disable dropout
    if eval_mode:
        rnn.eval()

    # disable gradient computation
    with torch.no_grad():
        cat, input_tensor, _ = tensorize(category, seed)
        hidden = rnn.init_hidden(num_layers=2)

        # add the length-1 batch dimension to match output from Dataset
        input_tensor = input_tensor.unsqueeze(0)

        # iterating through the max length
        output_name = seed
        for _ in range(max_len):
            out, hidden = rnn(input_tensor, cat, hidden)
            _, topi = out[:, -1, :].topk(
                1
            )  # grab top prediction for last character
            next_char = idx2char[int(topi.squeeze())]
            # break out of the loop if EOS is reached
            if break_on_eos and (next_char == "<EOS>"):
                break
            output_name += next_char
            input_tensor = topi

    # ensure training mode is (re-)enabled
    rnn.train()
    return output_name


# training
def train_step(
    cat_tensor: torch.LongTensor,
    input_tensor: torch.LongTensor,
    target_tensor: torch.LongTensor,
) -> float:
    optimizer.zero_grad()

    # forward pass
    hidden = rnn.init_hidden(num_layers=2)
    out, hidden = rnn(input_tensor, cat_tensor, hidden)
    loss = criterion(out[0, :, :], target_tensor[0, :])

    # back prop
    loss.backward()  # computes gradients w.r.t. and stores gradient values on parameters
    optimizer.step()  # is already aware of the parameters in the model, uses those gradients

    return loss.item()


if __name__ == "__main__":
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    # gather all the categories and names
    for filename in find_files("data/names/*.txt"):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines
    n_categories = len(all_categories)
    print(f"Total categories: {n_categories}")

    # instantiate dataset and dataloader class
    dataset = NameDataset(category_lines)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True
    )

    # input parameters for the network
    input_size = n_letters
    output_size = n_letters
    hidden_size = 128
    name_embeddings = 100
    cat_embeddings = 50

    # instantiate model
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
    print(rnn)

    # sample 10 names from untrained model
    for _ in range(10):
        seed = random.choice(string.ascii_letters)
        print(f"{seed} --> {sample(seed, category='English')}")

    # instantiate loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.0003)

    # actual training loop
    n_epochs = 2
    losses = []
    running_loss = 0.0
    best_loss = 100000
    for epoch in range(n_epochs):
        looper = tqdm.tqdm(data_loader, desc=f"epoch {epoch + 1}")
        for i, tensors in enumerate(looper):
            loss = train_step(*tensors)
            running_loss += loss
            if (i + 1) % 1000 == 0:
                losses.append(loss)
                looper.set_postfix({"Loss": running_loss / 1000.0})
                if (running_loss / 1000.0) < best_loss:
                    torch.save(rnn.state_dict(), "models/names.pt")
                    best_loss = running_loss / 1000.0
                running_loss = 0

    # plot the loss
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.show()
