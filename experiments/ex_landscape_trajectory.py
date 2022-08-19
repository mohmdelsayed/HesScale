import torch
from torch.optim import SGD, Adam
from optim_landscape.artificial_landscapes import RastriginLoss, RosenbrockLoss
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from hyperopt import fmin, hp, tpe
from backpack import backpack, extend
from backpack.extensions import DiagHessian, DiagGGNExact, DiagGGNMC

from hesscale import HesScale
from experiments.approximation_quality.optimizers.hesscale import HesScaleOptimizer
from experiments.approximation_quality.optimizers.ggn import GGNExactOptimizer
from experiments.approximation_quality.optimizers.ggn_mc import GGNMCOptimizer
from experiments.approximation_quality.optimizers.exact_diag_hess import ExactHessDiagOptimizer
from experiments.approximation_quality.optimizers.ada_hessian import Adahessian

def execute_steps(func, initial_state, optimizer_class, optimizer_config, num_iter=500):
    x = torch.Tensor(initial_state).requires_grad_(True)
    optimizer = optimizer_class([x], **optimizer_config)
    steps = []
    steps = np.zeros((2, num_iter + 1))
    steps[:, 0] = np.array(initial_state)
    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        f = func(x)
        with backpack(HesScale()):
            f.backward(create_graph=True)
        optimizer.step()
        steps[:, i] = x.detach().numpy()
    return steps


def objective_rastrigin(n_steps=100):
    def objective_rastrigin_(params):
        lr = params["lr"]
        optimizer_class = params["optimizer_class"]
        initial_state = (-2.0, 3.5)
        minimum = (0, 0)
        optimizer_config = dict(lr=lr)
        steps = execute_steps(
            extend(RastriginLoss()), initial_state, optimizer_class, optimizer_config, n_steps
        )
        return ((steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2)

    return objective_rastrigin_


def objective_rosenbrok(n_steps=100):
    def objective_rosenbrok_(params):
        lr = params["lr"]
        optimizer_class = params["optimizer_class"]
        minimum = (1.0, 1.0)
        initial_state = (-2.0, 2.0)
        optimizer_config = dict(lr=lr)
        steps = execute_steps(
            extend(RosenbrockLoss()), initial_state, optimizer_class, optimizer_config, n_steps
        )
        return ((steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2)

    return objective_rosenbrok_


def execute_experiments(
    optimizers, objective, func, plot_func, initial_state, seed=1, n_steps=500
):
    for item in optimizers:
        optimizer_class, lr_low, lr_hi = item
        space = {
            "optimizer_class": hp.choice("optimizer_class", [optimizer_class]),
            "lr": hp.loguniform("lr", lr_low, lr_hi),
        }
        best = fmin(
            fn=objective(n_steps=n_steps),
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            rstate=np.random.RandomState(seed),
        )
        print(best['lr'], optimizer_class)
        steps = execute_steps(
            func,
            initial_state,
            optimizer_class,
            {"lr": best["lr"]},
            num_iter=n_steps,
        )
        plot_func(steps, optimizer_class.__name__, best["lr"])


def plot_rastrigin(grad_iter, optimizer_name, lr):
    x = np.linspace(-4.5, 4.5, 250)
    y = np.linspace(-4.5, 4.5, 250)
    minimum = (0, 0)

    X, Y = np.meshgrid(x, y)
    Z = RastriginLoss(reduction="none")(torch.tensor(np.array([X, Y])))
    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 20, cmap="jet")
    ax.plot(iter_x, iter_y, color="r", marker="x")

    ax.set_title(
        "Rastrigin func: {} with "
        "{} iterations, lr={:.6}".format(optimizer_name, len(iter_x), lr)
    )
    plt.plot(*minimum, "gD")
    plt.plot(iter_x[-1], iter_y[-1], "rD")
    plt.savefig("rastrigin_{}.png".format(optimizer_name))
    # plt.show()


def plot_rosenbrok(grad_iter, optimizer_name, lr):
    x = np.linspace(-2, 2, 250)
    y = np.linspace(-1, 3, 250)
    minimum = (1.0, 1.0)

    X, Y = np.meshgrid(x, y)
    Z = RosenbrockLoss(reduction="none")(torch.tensor(np.array([X, Y])))

    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 90, cmap="jet")
    ax.plot(iter_x, iter_y, color="r", marker="x")

    ax.set_title(
        "Rosenbrock func: {} with {} "
        "iterations, lr={:.6}".format(optimizer_name, len(iter_x), lr)
    )
    plt.plot(*minimum, "gD")
    plt.plot(iter_x[-1], iter_y[-1], "rD")
    plt.savefig("rosenbrock_{}.png".format(optimizer_name))
    # plt.show()


optimizers = [
    (torch.optim.Adam, -6, -0.1),
    (torch.optim.SGD, -6, -0.5),
    (HesScaleOptimizer,  -6.0, -0.1),
    (Adahessian, -6, -0.0),
]
# execute_experiments(
#     optimizers,
#     objective_rastrigin,
#     extend(RastriginLoss()),
#     plot_func=plot_rastrigin,
#     initial_state=(-2.5, 3.0),
#     seed=678986,
#     n_steps=500,
# )

execute_experiments(
    optimizers,
    objective_rosenbrok,
    extend(RosenbrockLoss()),
    plot_func=plot_rosenbrok,
    initial_state=(-2.0, 2.0),
    seed=678986,
    n_steps=100,
)
