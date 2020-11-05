import torch


class Fra2EngDataset:
    def __init__(self, pair_vectors):
        self.pair_vectors = pair_vectors

    def __len__(self):
        return len(self.pair_vectors)

    def __getitem__(self, item):
        pair = self.pair_vectors[item]
        input_vector = pair[0]
        output_vector = pair[1]

        return torch.LongTensor(input_vector), torch.LongTensor(output_vector)
        # return {
        #     "input": torch.LongTensor(input_vector),
        #     "output": torch.LongTensor(output_vector),
        # }
