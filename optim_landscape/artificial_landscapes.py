from torch import nn as nn
# from torch import cos
# from math import pi
import torch
import math

class RastriginLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(RastriginLoss, self).__init__()
        self.reduction = reduction

    def forward(self, tensor):
        x, y = tensor[0], tensor[1]
        A = 10
        if self.reduction == "mean":
            return (
                A * 2 + (x ** 2 - A * torch.cos(x * math.pi * 2)) + (y ** 2 - A * torch.cos(y * math.pi * 2))
            ).mean()

        elif self.reduction == "sum":
            return (
                A * 2 + (x ** 2 - A * torch.cos(x * math.pi * 2)) + (y ** 2 - A * torch.cos(y * math.pi * 2))
            ).sum()
        elif self.reduction == "none":
            return (
                A * 2 + (x ** 2 - A * torch.cos(x * math.pi * 2)) + (y ** 2 - A * torch.cos(y * math.pi * 2))
            )
        else:
            raise Exception("{} is not a supported reduction".format(self.reduction))


class RosenbrockLoss(nn.Module):
    '''
    minimum = (0, 0) 
    '''
    def __init__(self, reduction="mean"):
        super(RosenbrockLoss, self).__init__()
        self.reduction = reduction

    def forward(self, tensor):
        x, y = tensor[0], tensor[1]

        if self.reduction == "mean":
            return (((1 - x) ** 2 + 100 * (y - x ** 2) ** 2)).mean()

        elif self.reduction == "sum":
            return ((1 - x) ** 2 + 100 * (y - x ** 2) ** 2).sum()
        elif self.reduction == "none":
            return ((1 - x) ** 2 + 100 * (y - x ** 2) ** 2)
        else:
            raise Exception("{} is not a supported reduction".format(self.reduction))