# dataset.py
import torch


class IMDBDataset:
    def __init__(self, reviews, targets):
        """
        :param reviews: numpy array containing review embeddings
        :param targets: labels, numpy array
        """
        self.reviews = reviews
        self.targets = targets

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review_vec = self.reviews[item, :]
        target = self.targets[item]
        return {
            "review": torch.tensor(review_vec, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float),
        }
