import random
import numpy as np
import torch
import torchvision
from torchvision import transforms
from .task import Task


class Cifar100(Task):
    """
    Iteratable Cifar100 task.
    Each sample is a 28x28 image and the label is a number between 0 and 9.
    """

    def __init__(self, name="cifar-100", subset='train', batch_size=128):
        assert subset in ['train', 'valid', 'test']
        self.subset = subset
        self.dataset = self.get_dataset()
        self.n_samples = len(self.dataset)
        self.step = 0
        # self.n_inputs = 784
        self.n_outputs = 100
        # self.change_freq = 5000
        self.criterion = "cross_entropy"
        if subset in ['train', 'valid']:
            indices = np.arange(len(self.dataset))
            random.shuffle(indices)
            valid_size = 10000
            self.train_indices = indices[:-valid_size]
            self.valid_indices = indices[-valid_size:]
            print(len(self.train_indices), len(self.valid_indices))
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

    def get_dataset(self):
        transform_augmented = transforms.Compose([
            transforms.Pad(padding=2),
            transforms.RandomCrop(size=(32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
            transforms.ToTensor(),
            transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))
        ])
        transform_not_augmented = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))
        ])
        transform = transform_augmented if self.subset == 'train' else transform_not_augmented
        # transform = transform_not_augmented

        train = self.subset in ['train', 'valid']
        dataset = torchvision.datasets.CIFAR100(
            "dataset",
            train=train,
            download=True,
            transform=transform,
        )

        return dataset

    def get_dataloader(self, dataset):
        sampler = None
        if self.subset == 'train':
            sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_indices)
        elif self.subset == 'valid':
            sampler = torch.utils.data.sampler.SubsetRandomSampler(self.valid_indices)
        elif self.subset == 'test':
            sampler = torch.utils.data.sampler.SubsetRandomSampler(np.arange(len(dataset)))

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            # shuffle=True,
            sampler=sampler,
        )

if __name__ == "__main__":
    task = Cifar100()
    for i, (x, y) in enumerate(task):
        print(x.shape, y.shape)
        if i == 10:
            break
