import unicodedata
import re
import random
from collections import defaultdict

import torch

# Global variables to be used everywhere
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

eng_prefixes = (
    "i am ",
    "i m ",
    "he is",
    "he s ",
    "she is",
    "she s ",
    "you are",
    "you re ",
    "we are",
    "we re ",
    "they are",
    "they re ",
)

# Class to create look-up tables for the vocabulary
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2idx = defaultdict(int)
        self.word2count = defaultdict(int)
        self.idx2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # count for EOS and SOS

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.word2count[word] = 1
            self.idx2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Change unicode characters to ASCII
def unicode_to_ascii(word):
    word = "".join(
        c
        for c in unicodedata.normalize("NFD", word)
        if unicodedata.category(c) != "Mn"
    )
    return word


# Lowercase everything and remove punctuation
def normalize_word(word):
    word = unicode_to_ascii(word.lower().strip())
    word = re.sub(r"([.!?])", r" \1", word)
    word = re.sub(r"[.!?]", r"", word)
    word = re.sub(r"[^a-zA-Z]+", r" ", word)
    word = word.strip()
    return word


# Read the input file and generate sentence pairs
def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # read the file
    lines = (
        open(f"data/{lang1}-{lang2}.txt", encoding="utf-8")
        .read()
        .strip()
        .split("\n")
    )

    # split every line into pair of sentences and normalize
    pairs = [
        [normalize_word(word) for word in line.split("\t")] for line in lines
    ]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)

    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filter_pair(p):
    return (
        len(p[0].split(" ")) < MAX_LENGTH
        and len(p[1].split(" ")) < MAX_LENGTH
        and p[1].startswith(eng_prefixes)
    )


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print(f"Read {len(pairs)} sentence pairs")
    pairs = filter_pairs(pairs)
    print(f"Filtered to {len(pairs)} sentence pairs")
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    print("Counted words:")
    print("\t", input_lang.name + ":", input_lang.n_words)
    print("\t", output_lang.name + ":", output_lang.n_words)
    return input_lang, output_lang, pairs


def vector_from_sentence(lang, sentence):
    return [lang.word2idx[word] for word in sentence.split(" ")]


def vector_from_pair(input_lang, output_lang, pair):
    input_vector = vector_from_sentence(input_lang, pair[0])
    output_vector = vector_from_sentence(output_lang, pair[1])
    input_vector.append(EOS_token)
    output_vector.append(EOS_token)

    return [input_vector, output_vector]


# if __name__ == "__main__":
#     input_lang, output_lang, pairs = prepare_data("eng", "fra", True)
#     print(random.choice(pairs))

#     sentence_vectors = []
#     for pair in pairs:
#         vectors = vector_from_pair(pair)
#         sentence_vectors.append(vectors)
#     print(sentence_vectors[2356:2366])
