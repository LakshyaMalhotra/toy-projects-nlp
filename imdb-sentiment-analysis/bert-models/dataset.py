# dataset.py
import config
import torch


class BERTDataset:
    def __init__(self, reviews, targets):
        """
        :param reviews: list or numpy array of strings
        :param targets: list or numpy array of the sentiments (binary)
        """
        self.reviews = reviews
        self.targets = targets

        # fetch max_len and tokenizer from config file
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        # this returns the length of the dataset
        return len(self.reviews)

    def __getitem__(self, item):
        # for a given item index, return a dictionary of the inputs
        review = str(self.reviews[item])
        review = " ".join(review.split())

        # encode_plus comes from huggingface's transformers and exists for all
        # tokenizers they offer.
        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        # ids are the ids of tokens generated after tokenization
        ids = inputs["input_ids"]

        # mask is 1 where we have input and 0 where we have padding
        mask = inputs["attention_mask"]

        # token type ids behave the same way as mask in this case
        token_type_ids = inputs["token_type_ids"]

        # pad the sequence
        # padding_length = self.max_len - len(ids)
        # ids = ids + ([0] * padding_length)
        # mask = mask + ([0] * padding_length)
        # token_type_ids = token_type_ids + ([0] * padding_length)

        # return everything
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[item], dtype=torch.float),
        }
