import torch
import torchvision
from .task import Task


class StationaryMNIST(Task):
    """
    Iteratable MNIST task.
    Each sample is a 28x28 image and the label is a number between 0 and 9.
    """

    def __init__(self, name="stationary_mnist", batch_size=8):
        self.dataset = self.get_dataset(True)
        self.step = 0
        self.n_inputs = 784
        self.n_outputs = 10
        self.change_freq = 5000
        self.criterion = "cross_entropy"
        super().__init__(name, batch_size)

    def __next__(self):
        self.step += 1
        try:
            # Samples from dataset
            return next(self.iterator)
        except StopIteration:
            # restart the iterator if the previous iterator is exhausted.
            self.iterator = self.generator()
            return next(self.iterator)

    def generator(self):
        return iter(self.get_dataloader(self.dataset))

    def get_dataset(self, train=True):
        return torchvision.datasets.MNIST(
            "dataset",
            train=train,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,)),
                    torchvision.transforms.Lambda(lambda x: torch.flatten(x)),
                ]
            ),
        )

    def get_dataloader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

if __name__ == "__main__":
    task = StationaryMNIST()
    for i, (x, y) in enumerate(task):
        print(x.shape, y.shape)
        if i == 10:
            break
