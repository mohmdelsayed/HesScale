import numpy as np


class TargetGenerator:
    def __init__(
        self,
        in_size=32,
        out_size=2,
        batch_size=1,
        dataset_size=10000,
    ):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.out_size = out_size
        self.in_size = in_size

    def get_example(self, batch_size=1):
        example = np.random.randn(batch_size, self.in_size)
        labels = np.random.randint(self.out_size, size=batch_size)
        return example, labels

    def get_dataset(self, dataset_size=1000):
        example = np.random.randn(dataset_size, self.in_size)
        labels = np.random.randint(self.out_size, size=dataset_size)
        return example, labels
