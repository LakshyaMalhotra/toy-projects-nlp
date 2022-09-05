import torch
from typing import List, Tuple

TensorPair = Tuple[torch.LongTensor, torch.LongTensor]


class NameDataset:
    def __init__(
        self, all_categories: List[str], names: List[tuple], char2idx: dict
    ):
        self.names = names
        self.all_categories = all_categories
        self.char2idx = char2idx

    def __len__(self):
        return len(self.names)

    def __getitem__(self, item: int) -> TensorPair:
        data = self.names[item]
        category = torch.tensor(
            [self.all_categories.index(data[0])], dtype=torch.long
        )
        name = data[1]
        name = torch.tensor(
            [self.char2idx[letter] for letter in name], dtype=torch.long
        )

        return name, category

