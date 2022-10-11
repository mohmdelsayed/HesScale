import torch
from backpack import backpack, extend
from hesscale_optimizers import AdaHesScale, AdaHesScaleGGN


n_obs = (512, 512)
n_channels = 3
n_classes = 10
batch_size = 13
nhidden = 8
lr = 0.0004
T = 1

model = torch.nn.Sequential(
    torch.nn.Conv2d(n_channels, nhidden, 2),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(8, 8),
    torch.nn.Conv2d(nhidden, nhidden, 2),
    torch.nn.ReLU(),
    torch.nn.AvgPool2d(8, 8),
    torch.nn.Flatten(),
    torch.nn.Linear(392, n_classes),
)


loss_func = torch.nn.CrossEntropyLoss()
optimizer = AdaHesScale(model.parameters(), lr=lr)
# optimizer = AdaHesScaleGGN(model.parameters(), lr=lr)

savefield = optimizer.method.savefield

extend(model)
extend(loss_func)

for i in range(T):
    inputs = torch.randn((batch_size, n_channels, n_obs[0], n_obs[1]))
    target_class = torch.randint(0, n_classes, (batch_size,))

    prediction = model(inputs)
    optimizer.zero_grad()
    loss = loss_func(prediction, target_class)

    with backpack(optimizer.method):
        loss.backward()

    optimizer.step()

    for (name, param) in model.named_parameters():       
        print('g:', name, getattr(param, 'grad').shape) 
        print('h:', name, getattr(param, savefield).shape)
