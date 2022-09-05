# utils.py
import glob
import os

import unicodedata
import string

# vocabulary of all the letters and some special characters
all_letters = string.ascii_letters + ".,'"
n_letters = len(all_letters)

# look up tables for index to char and vice-versa
idx2char = dict(enumerate(all_letters))
char2idx = {v: k for k, v in idx2char.items()}

# create entire preprocessing pipeline
class PreProcess:
    def __init__(self):
        # containers to hold names and categories
        self.all_letters = all_letters
        self.n_letters = n_letters
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.all_categories = []
        self.all_names = []
        self.file_names = None

    # find all the files
    def find_files(self, path):
        self.file_names = glob.glob(path)

    # Turn a Unicode string to plain ASCII,
    # thanks to https://stackoverflow.com/a/518232/2809427
    @staticmethod
    # thanks to: https://stackoverflow.com/a/23508293/7165280
    def unicode_to_ascii(s):
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn" and c in all_letters
        )

    # read the lines from the files
    # thanks to: https://stackoverflow.com/a/35459657/7165280
    @staticmethod
    def read_lines(filename):
        lines = open(filename, encoding="utf-8").read().strip().split("\n")
        return [PreProcess.unicode_to_ascii(name) for name in lines]

    # read the lines from a single file
    def read_file(self, file_name):
        category = os.path.splitext(os.path.basename(file_name))[0]
        category_names = self.read_lines(file_name)
        names = zip([category] * len(category_names), category_names)
        self.all_categories.append(category)
        self.all_names.extend(list(names))

    # read all the files
    def read_files(self):
        for file_name in self.file_names:
            self.read_file(file_name)

    def read_and_preprocess(self, path):
        # print("Finding all the files")
        self.find_files(path)
        # print("Now reading...")
        self.read_files()

