import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class TargetGenerator:
    def __init__(
        self,
        task,
        in_size=32,
        out_size=2,
        hidden_units=32,
        batch_size=1,
        noise=1.0,
        dataset_size=10000,
    ):
        self.info = (in_size, out_size)
        self.batch_size = batch_size
        self.target_noise = noise
        self.dataset_size = dataset_size
        if task == "classification":
            self.target = TargetNetClassification(
                self.info[1], self.info[0], hidden_units
            )
        elif task == "regression":
            self.target = TargetNetRegression(self.info[1], self.info[0], hidden_units)
        elif task == "mnist":
            self.target = iter(InfiniteMNIST())
        else:
            raise ("No valid task")

    def get_example(self):
        if isinstance(self.target, TargetNetRegression):
            example = np.random.randn(self.batch_size, self.info[0])
            noise = np.random.randn(self.batch_size, self.info[1]).astype("f")
            target = self.target(example).detach().numpy() + noise * self.target_noise
            return example, target
        elif isinstance(self.target, TargetNetClassification):
            example = np.random.randn(self.batch_size, self.info[0])
            noise = np.random.randn(self.batch_size, self.info[1]).astype("f")
            target = self.target(example).detach().numpy() + noise * self.target_noise
            labels = np.argmax(target, axis=1)
            return example, labels
        elif isinstance(self.target, InfiniteMNIST):
            return next(self.target)
        else:
            raise ("iid task is not defined.")

    def get_dataset(self, dataset_size=1000):
        if isinstance(self.target, TargetNetRegression):
            example = np.random.randn(dataset_size, self.info[0])
            noise = np.random.randn(dataset_size, self.info[1]).astype("f")
            target = self.target(example).detach().numpy() + noise * self.target_noise
            return example, target
        elif isinstance(self.target, TargetNetClassification):
            example = np.random.randn(dataset_size, self.info[0])
            noise = np.random.randn(dataset_size, self.info[1]).astype("f")
            target = self.target(example).detach().numpy() + noise * self.target_noise
            labels = np.argmax(target, axis=1)

            # example = np.random.randn(dataset_size, 1, self.info[0], self.info[0])
            example = np.random.randn(dataset_size, self.info[0])

            return example, labels
        else:
            raise ("iid task is not defined.")


class TargetNetClassification(nn.Module):
    def __init__(
        self,
        n_classes,
        n_obs,
        hidden_units,
    ):
        super(TargetNetClassification, self).__init__()
        self.inner = nn.Linear(n_obs, hidden_units, bias=False)
        self.outer = nn.Linear(hidden_units, n_classes, bias=False)

        torch.nn.init.normal_(self.inner.weight, mean=2.0, std=1.0)
        torch.nn.init.normal_(self.outer.weight, mean=0.0, std=1.0)

    def forward(self, x):
        x = torch.from_numpy(x).float()
        x = torch.tanh(self.inner(x))
        x = self.outer(x)
        # x = F.softmax(x, dim=1)
        return x


class TargetNetRegression(nn.Module):
    def __init__(
        self,
        n_classes,
        n_obs,
        hidden_units,
    ):
        super(TargetNetRegression, self).__init__()
        self.inner = nn.Linear(n_obs, hidden_units, bias=False)
        self.outer = nn.Linear(hidden_units, n_classes, bias=False)

        torch.nn.init.normal_(self.inner.weight, mean=2.0, std=1.0)
        torch.nn.init.normal_(self.outer.weight, mean=0.0, std=1.0)

    def forward(self, x):
        x = torch.from_numpy(x).float()
        x = torch.tanh(self.inner(x))
        x = self.outer(x)
        return x
