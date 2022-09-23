# Adapted from https://github.com/jettify/pytorch-optimizer
import torch
from experiments.optim_landscape.artificial_landscapes import RosenbrockLoss
import matplotlib.pyplot as plt
import numpy as np
import torch
from hyperopt import fmin, hp, tpe
from backpack import backpack, extend

from hesscale import HesScale
from experiments.computational_cost.optimizers.ada_hesscale import HesScaleOptimizer

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
            f.backward()
        optimizer.step()
        steps[:, i] = x.detach().numpy()
    return steps

def objective_rosenbrok(n_steps=100):
    def objective_rosenbrok_(params):
        lr = params["lr"]
        optimizer_class = params["optimizer_class"]
        minimum = (1.0, 1.0)
        initial_state = (-2.0, 2.0)
        optimizer_config = dict(lr=lr)
        steps = execute_steps(
            extend(RosenbrockLoss()),
            initial_state,
            optimizer_class,
            optimizer_config,
            n_steps,
        )
        return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2

    return objective_rosenbrok_


def execute_experiments(
    optimizers, objective, func, plot_func, initial_state, seed=1, n_steps=500
):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    x = np.linspace(-3, 2, 250)
    y = np.linspace(-2, 6, 250)
    minimum = (1.0, 1.0)

    X, Y = np.meshgrid(x, y)
    Z = RosenbrockLoss(reduction="none")(torch.tensor(np.array([X, Y])))

    ax.contour(X, Y, Z, 90, cmap="jet")

    ax.set_title("Rosenbrock func")

    TABLEAU_COLORS = {
        "tab:blue": "#1f77b4",
        "tab:orange": "#ff7f0e",
        "tab:green": "#2ca02c",
        "tab:red": "#d62728",
        "tab:purple": "#9467bd",
        "tab:brown": "#8c564b",
        "tab:pink": "#e377c2",
        "tab:gray": "#7f7f7f",
        "tab:olive": "#bcbd22",
        "tab:cyan": "#17becf",
    }

    for item, color in zip(optimizers, TABLEAU_COLORS):
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

        steps = execute_steps(
            func,
            initial_state,
            optimizer_class,
            {"lr": best["lr"]},
            num_iter=n_steps,
        )

        plot_func(ax, steps, optimizer_class.__name__, best["lr"], color, minimum)
    ax.legend()
    plt.savefig("rastrigin_{}.pdf".format(optimizer_class.__name__))


def plot_rosenbrok(ax, grad_iter, optimizer_name, lr, color, minimum):
    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]
    ax.plot(iter_x, iter_y, marker=".", color=color, label=optimizer_name+' ' +str(lr))
    ax.plot(*minimum, "gD")
    ax.plot(iter_x[-1], iter_y[-1], "D", color=color)


optimizers = [
    (torch.optim.Adam, -4.0, -0.1),
    (torch.optim.SGD, -6.0, -0.1),
    (HesScaleOptimizer, -4.0, -0.1),
]

execute_experiments(
    optimizers,
    objective_rosenbrok,
    extend(RosenbrockLoss()),
    plot_func=plot_rosenbrok,
    initial_state=(-2.0, 2.0),
    seed=12,
    n_steps=499,
)
