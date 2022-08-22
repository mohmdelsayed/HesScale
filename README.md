# HesScale: Computation of Hessian Diagonals in a Scalable Way


HesScale is build on top of Pytorch and BackPack. It allows for Hessian diagonals to backpropagate through the layers of the network.

## Installation:
#### 1. You need to have environemnt with python 3.7:
``` sh
python3.7 -m venv .hesscale
source .hesscale/bin/activate
```
#### 2. Install Dependencies:
```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pip install .
```

## Run a minimal example:
We added a couple of minimal examples in the `examples` directory for easier understanding of how to use this package. Here is a minimal example:
```python
#!/usr/bin/env python3
import torch
from hesscale import HesScale
from backpack import backpack, extend

hidden_units = 128
n_obs = 6
n_classes = 10
lr = 0.0004
batch_size = 1

model = torch.nn.Sequential(
    torch.nn.Linear(6, 128),
    torch.nn.Tanh(),
    torch.nn.Linear(128, 10),
)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

extend(model)
extend(loss_func)

inputs = torch.randn((batch_size, n_obs))
target_class = torch.randint(0, n_classes, (batch_size,))

prediction = model(inputs)
optimizer.zero_grad()
loss = loss_func(prediction, target_class)

with backpack(HesScale()):
    loss.backward()

optimizer.step()

for (name, param) in model.named_parameters():
    print(name, param.hesscale.shape)
```

## Contributing
We appreciate any help to extend HesScale to recurrent neural networks. If you consider contributing, please fork the repo and create a pull request. 

## License
Distributed under the [MIT License](https://opensource.org/licenses/MIT). See `LICENSE` for more information.


## How to cite
If you use our code, please consider citing our paper too.
```
@inproceedings{elsayed2022hesscale,
    title     = {HesScale: Computation of Hessian Diagonals in a Scalable Way,
    author    = {Mohamed Elsayed and Rupam Mahmood},
    booktitle = {},
    url       = {},
    year      = {2022}
}
```
