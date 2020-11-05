import torch


class Fra2EngDataset:
    def __init__(self, pair_vectors, device=torch.device("cpu")):
        self.pair_vectors = pair_vectors
        self.device = device

    def __len__(self):
        return len(self.pair_vectors)

    def __getitem__(self, item):
        pair = self.pair_vectors[item]
        input_vector = pair[0]
        output_vector = pair[1]

        return {
            "input": torch.LongTensor(input_vector, device=self.device),
            "output": torch.LongTensor(output_vector, device=self.device),
        }

