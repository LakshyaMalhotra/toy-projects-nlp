# Library imports
import random
import time
import glob
import os
import string
import unicodedata

import torch

# find all the files
def find_files(path: str) -> list:
    return glob.glob(path)


# turn unicode chars to plain ascii
# strips off the accent etc
def unicode_to_ascii(word: str) -> str:
    word = "".join(
        c
        for c in unicodedata.normalize("NFD", word)
        if unicodedata.category(c) != "Mn" and (c in all_letters)
    )
    return word


# one hot tensor for category
def category_tensor(category: str) -> torch.Tensor:
    li = all_categories.index(category)
    cat_tensor = torch.zeros(1, n_categories)
    cat_tensor[0][li] = 1
    return cat_tensor


# create a one-hot tensor for each letter in each line
def input_tensor(line: str) -> torch.Tensor:
    line_tensor = torch.zeros(len(line), 1, n_letters)
    # letter_id = [all_letters.find(letter) for letter in line]
    for li in range(len(line)):
        letter = line[li]
        line_tensor[li][0][all_letters.find(letter)] = 1
    return line_tensor


# long tensor of second letter to EOS for target
def target_tensor(line: str) -> torch.Tensor:
    letter_id = [all_letters.find(line[i]) for i in range(1, len(line))]
    letter_id.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_id)


# generate random training examples
def random_choice(lst: list) -> torch.Tensor:
    return lst[random.randint(0, len(lst) - 1)]


def random_training_example():
    category = random_choice(all_categories)
    cat_tensor = category_tensor(category)
    line = random_choice(category_lines[category])
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return category, line, cat_tensor, input_line_tensor, target_line_tensor


# read the files and split into lines and normalize them
def read_lines(filename: str) -> list:
    lines = open(filename, "r", encoding="utf-8").read().split("\n")
    return [unicode_to_ascii(line) for line in lines]


if __name__ == "__main__":
    # read all the files
    files = find_files("data/names/*.txt")
    print(files)

    # vocabulary
    all_letters = string.ascii_letters + ".,;'"
    print(f"Letters: {all_letters}")

    # total number of letters
    n_letters = len(all_letters) + 1  # for EOS
    print(f"Total number of letters: {n_letters}")

    # check if the word normalization is working properly
    print(unicode_to_ascii("Schrödinger"))

    # Build a category line dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    # print(read_lines("data/names/German.txt"))

    letters_list = []
    for file in files:
        category = os.path.splitext(os.path.basename(file))[0]
        all_categories.append(category)
        lines = read_lines(file)
        category_lines[category] = lines
        for name in lines:
            for c in name:
                letters_list.append(c)
    n_categories = len(all_categories)
    print(f"Total letters: {len(letters_list)}")
    print(f"Number of unique letters: {len(set(letters_list))}")
    print(f"Categories: {all_categories}")

    print("Generating a random training example...")
    (
        cat,
        line,
        cat_tensor,
        in_line_tensor,
        target_line_tensor,
    ) = random_training_example()

    print(f"Category: {cat}")
    print(f"Word: {line}")
    print(f"Category tensor: {cat_tensor}")
    print(f"Target line tensor: {target_line_tensor}")
    print(f"Input line tensor shape: {in_line_tensor.shape}")
