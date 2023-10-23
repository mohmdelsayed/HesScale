# HesScale: Scalable Computation of Hessian Diagonals


HesScale is built on top of Pytorch and BackPack. It allows for Hessian diagonals to backpropagate through the layers of the network.

## Installation:
#### 1. You need to have environemnt with python 3.7:
``` sh
python3.7 -m venv .hesscale
source .hesscale/bin/activate
```
#### 2. Install Dependencies:
```sh
python -m pip install --upgrade pip
pip install .
```

## Run an optimization experiment:
You need to create a task in `core/task` and write how to run in in `core/run`. You can also add new optimizers in `core/learner` or networks in `core/netowrk`. After that, you can create your experiment script in `experiments/` that specify the task, compared learners, grid search and how to run the task. Here is an example:

```python
python ex1_stationary_mnist.py 
```

which will generate a `generated_cmds` directory containing text files that has the python cmds need to be run. In addition, there will be a script generated automatically for running on compute canada.


## Run an optimization experiment:
You can run 

```python
python experiments/approximation_quality/run.py 
```

## Run a minimal example:
We added a couple of minimal examples in the `examples` directory for easier understanding of how to use this package. Here is a minimal example:
```python
import torch
from backpack import backpack, extend
from optimizers.adahesscale import AdaHesScale

hidden_units = 128
n_obs = 6
n_classes = 10
lr = 0.0004
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

extend(model)
extend(loss_func)

inputs = torch.randn((batch_size, n_obs))
target_class = torch.randint(0, n_classes, (batch_size,))

prediction = model(inputs)
optimizer.zero_grad()
loss = loss_func(prediction, target_class)

with backpack(optimizer.method):
    loss.backward()

optimizer.step()

```

## Contributing
We appreciate any help to extend HesScale to recurrent neural networks. If you consider contributing, please fork the repo and create a pull request. 

## License
Distributed under the [MIT License](https://opensource.org/licenses/MIT). See `LICENSE` for more information.

## Reproduction
To reproduce the experiments in the paper, you run the scripts in `experiemnts` directory for reproducing the approximation-quality experiment and the computational-cost experiment. For reproducing the training plots, please refer to the our other [repo](https://github.com/mohmdelsayed/HesScale-Comparisons) that uses [DeepOBS](https://github.com/fsschneider/DeepOBS) for complete reproduction of our optimization results.

## How to cite
If you use our code, please consider citing our paper too.
```
Elsayed, M., & Mahmood, A. R. (2022). HesScale: Scalable Computation of Hessian Diagonals. arXiv preprint arXiv:2210.11639.
```
