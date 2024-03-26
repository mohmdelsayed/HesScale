import torch
import torchvision
from .task import Task


class Cifar100(Task):
    """
    Iteratable Cifar100 task.
    Each sample is a 28x28 image and the label is a number between 0 and 9.
    """

    def __init__(self, name="cifar-100", train=True, batch_size=128):
        self.dataset = self.get_dataset(train)
        self.n_samples = len(self.dataset)
        self.step = 0
        # self.n_inputs = 784
        self.n_outputs = 100
        # self.change_freq = 5000
        self.criterion = "cross_entropy"
        super().__init__(name, batch_size)

    def __next__(self):
        self.step += 1
        try:
            # Samples from dataset
            return next(self.iterator)
        except StopIteration:
            return None, None
    
    def generator(self):
        return iter(self.get_dataloader(self.dataset))

    def get_dataset(self, train=True):
        return torchvision.datasets.CIFAR100(
            "dataset",
            train=train,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                    # torchvision.transforms.Lambda(lambda x: torch.flatten(x)),
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
    task = Cifar100()
    for i, (x, y) in enumerate(task):
        print(x.shape, y.shape)
        if i == 10:
            break
