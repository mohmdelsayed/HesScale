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

The above command should install the correct versions of gym and mujoco-py packages that run on compute canada. To run experiments on mujoco environments, please copy the [binaries for mujoco version 2.1.0](https://github.com/google-deepmind/mujoco/releases/tag/2.1.0) to `~/.mujoco/mujoco210/`.

## Run an experiment:
You need to create a task in `core/task` and write how to run in in `core/run`. You can also add new optimizers in `core/learner` or networks in `core/netowrk`. After that, you can create your experiment script in `experiments/` that specify the task, compared learners, grid search and how to run the task. Here is an example:

```python
python experiments/ex1_stationary_mnist.py 
```

which will generate a `generated_cmds` directory containing text files that has the python cmds need to be run. In addition, there will be a script generated automatically for running on compute canada.


## License
Distributed under the [MIT License](https://opensource.org/licenses/MIT). See `LICENSE` for more information.

## Reproduction
To reproduce the experiments in the paper, you run the scripts in `experiemnts` directory for reproducing the approximation-quality experiment and the computational-cost experiment. For reproducing the training plots, please refer to the our other [repo](https://github.com/mohmdelsayed/HesScale-Comparisons) that uses [DeepOBS](https://github.com/fsschneider/DeepOBS) for complete reproduction of our optimization results.

## How to cite
If you use our code, please consider citing our paper too.
```
Elsayed, M., & Mahmood, A. R. (2022). HesScale: Scalable Computation of Hessian Diagonals. arXiv preprint arXiv:2210.11639.
```
