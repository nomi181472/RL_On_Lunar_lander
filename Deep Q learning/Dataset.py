# dataset
from torch.utils.data.dataset import IterableDataset, T_co


class RLDataset(IterableDataset):

    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, buffer, sample_size=2000):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        for experience in self.buffer.sample(self.sample_size):
            yield experience
