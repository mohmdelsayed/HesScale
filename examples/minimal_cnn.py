from backpack import backpack, extend
from hesscale import HesScale
import torch.nn as nn
import torch

n_obs = (512, 512)
n_channels = 3
n_classes = 10
batch_size = 13
nhidden = 8
lr = 0.0004
T = 1

model = nn.Sequential(
    nn.Conv2d(n_channels, nhidden, 5),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(nhidden, nhidden, 5),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(nhidden, nhidden, 5),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(nhidden, nhidden, 5),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(nhidden, nhidden, 5),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(nhidden, nhidden, 5),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(128, n_classes),
)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

extend(model)
extend(loss_func)

for i in range(T):
    inputs = torch.randn((batch_size, n_channels, n_obs[0], n_obs[1]))
    target_class = torch.randint(0, n_classes, (batch_size,))

    prediction = model(inputs)
    optimizer.zero_grad()
    loss = loss_func(prediction, target_class)

    with backpack(HesScale()):
        loss.backward()

    optimizer.step()

    for (name, param) in model.named_parameters():
        print(name, param.hesscale.shape)
