# utils.py
# Library imports
import random
import warnings

warnings.filterwarnings("ignore")

import torch
from torchtext.datasets import Multi30k

from torchtext.data import Field, BucketIterator
import spacy


# SEED everything
# SEED = 23
# random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(SEED)
#     torch.backends.cudnn.deterministic = True

# load the language models
spacy_en = spacy.load("en_core_web_sm")
spacy_de = spacy.load("de_core_news_sm")

# class to do the basic preprocessing
class Preprocess(Field, BucketIterator, Multi30k):
    def __init__(
        self, batch_size=64, verbose=False, device=torch.device("cpu")
    ):
        super(Preprocess, self).__init__()
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = device
        self.source = Field(
            tokenize=self.tokenize_de,
            init_token="<sos>",
            eos_token="<eos>",
            lower=True,
        )
        self.target = Field(
            tokenize=self.tokenize_en,
            init_token="<sos>",
            eos_token="<eos>",
            lower=True,
        )

    @staticmethod
    def tokenize_de(text):
        """
        Tokenizes text from a string and create a list of tokens after reversing it.
        """
        return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

    @staticmethod
    def tokenize_en(text):
        """
        Tokenizes text from a string and create a list of tokens.
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def get_dataset(self):
        self.train_data, self.valid_data, _ = Multi30k.splits(
            exts=(".de", ".en"), fields=(self.source, self.target)
        )
        self.source.build_vocab(self.train_data, min_freq=2)
        self.target.build_vocab(self.train_data, min_freq=2)
        self.input_size = len(self.source.vocab)
        self.target_size = len(self.target.vocab)

        if self.verbose:
            print(
                f"Number of training examples: {len(self.train_data.examples)}"
            )
            print(
                f"Number of validation examples: {len(self.valid_data.examples)}"
            )
            print("Vocabularies created...")
            print(f"Source vocabulary size: {self.input_size}")
            print(f"Target vocabulary size: {self.target_size}")

    def batchify(self):
        self.train_iterator, self.valid_iterator = BucketIterator.splits(
            (self.train_data, self.valid_data),
            batch_size=self.batch_size,
            device=self.device,
        )
        return self.train_iterator, self.valid_iterator


if __name__ == "__main__":
    pp = Preprocess(verbose=True, batch_size=8)
    # create datasets and vocabularies
    pp.get_dataset()
    train_it, valid_it = pp.batchify()
    count = 0
    for batch in train_it:
        print(f"Source tensor: {batch.src}")
        print(f"Target tensor: {batch.trg}")
        count += 1
        print(count)
