import torch
from backpack import backpack, extend
from optimizers.adahesscale import AdaHesScale
# from optimizers.adahesscaleggn import AdaHesScaleGGN


hidden_units = 128
n_obs = 6
n_classes = 10
lr = 0.0004
T = 1
batch_size = 1

model = torch.nn.Sequential(
    torch.nn.Linear(n_obs, hidden_units),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_units, hidden_units),
    torch.nn.Tanh(),
    torch.nn.Linear(hidden_units, n_classes),
)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = AdaHesScale(model.parameters(), lr=lr)
# optimizer = AdaHesScaleGGN(model.parameters(), lr=lr)

savefield = optimizer.method.savefield

extend(model)
extend(loss_func)

for i in range(T):
    inputs = torch.randn((batch_size, n_obs))
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
