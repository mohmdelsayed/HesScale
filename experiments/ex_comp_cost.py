import time, math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from backpack import backpack, extend
from torch import nn
import warnings

warnings.filterwarnings("ignore")
from experiments.computational_cost.optimizers.ada_hessian import Adahessian
from experiments.computational_cost.optimizers.exact_diag_hess import (
    ExactHessDiagOptimizer,
)
from experiments.computational_cost.optimizers.ggn import GGNExactOptimizer
from experiments.computational_cost.optimizers.ggn_mc import GGNMCOptimizer
from experiments.computational_cost.optimizers.hesscale import HesScaleOptimizer
from experiments.computational_cost.optimizers.kfac import KFACOptimizer


def get_optimizer(lnet, methods, method_name):
    if method_name == "sgd":
        opt = methods[method_name](lnet.parameters(), lr=1e-4)
        opt_class_backpack = None
    elif method_name == "adam":
        opt = methods[method_name](lnet.parameters(), betas=[0, 0.999])
        opt_class_backpack = None
    elif method_name == "adahess":
        opt = methods[method_name](lnet.parameters(), betas=[0, 0.999])
        opt_class_backpack = None
    elif method_name == "hesscale":
        opt = methods[method_name](lnet.parameters(), betas=[0, 0.999])
        opt_class_backpack = methods[method_name]
    elif method_name == "ggn":
        opt = methods[method_name](lnet.parameters(), betas=[0, 0.999])
        opt_class_backpack = methods[method_name]
    elif method_name == "ggnmc":
        opt = methods[method_name](lnet.parameters(), betas=[0, 0.999])
        opt_class_backpack = methods[method_name]
    elif method_name == "h":
        opt = methods[method_name](lnet.parameters(), betas=[0, 0.999])
        opt_class_backpack = methods[method_name]
    elif method_name == "kfac":
        opt = methods[method_name](lnet.parameters(), betas=[0, 0.999])
        opt_class_backpack = methods[method_name]
    else:
        raise "Method is not supported"
    return opt, opt_class_backpack


def main():
    np.random.seed(1234)
    torch.manual_seed(1234)

    methods = {
        "hesscale": HesScaleOptimizer,
        "adahess": Adahessian,
        "ggn": GGNExactOptimizer,
        "ggnmc": GGNMCOptimizer,
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "kfac": KFACOptimizer,
        "h": ExactHessDiagOptimizer,
    }

    all_means = []
    all_stds = []
    T = 1
    option_quadratic = True
    list_range = (
        [1024, 2048, 4096, 8192, 16384, 32768]
        if option_quadratic
        else [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    )

    for output in list_range:

        means = []
        stds = []

        if option_quadratic:
            ninpts = 16
            layers = [
                nn.Linear(1, ninpts, bias=False),
                nn.Linear(ninpts, output, bias=False),
            ]
            net = nn.Sequential(*layers)
        else:
            ninpts = 1
            nhidden = 1024
            layers = [nn.Linear(ninpts, nhidden, bias=False)]
            for i in range(output):
                layers.append(nn.Linear(nhidden, nhidden, bias=False))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(nhidden, 100, bias=False))
            net = nn.Sequential(*layers)
            output = 100
        print("Number of params:", sum(p.numel() for p in net.parameters()))

        for method_name in methods:

            opt, opt_class_backpack = get_optimizer(net, methods, method_name)

            if opt_class_backpack is not None:
                extend(net)
                lossf = extend(nn.MSELoss())
            else:
                lossf = nn.MSELoss()

            # Experiments
            mses = np.zeros((1, T))
            loop_times = np.zeros(T)

            for t in range(T):
                x = torch.randn((1, 1))
                y = torch.randn((1, output))
                loss = lossf(net(x), y)
                opt.zero_grad()

                prev_t = time.perf_counter()

                if opt_class_backpack is not None:
                    with backpack(opt_class_backpack.method):
                        loss.backward()
                else:
                    loss.backward(create_graph=True)

                opt.step()

                loop_times[t] = time.perf_counter() - prev_t
                mses[0, t] = loss.item()

            means.append(loop_times.mean())
            stds.append(2.0 * loop_times.std() / math.sqrt(T))

        all_means.append(means)
        all_stds.append(stds)

    all_means_array = np.asarray(all_means)
    all_stds_array = np.asarray(all_stds)

    for mean, std in zip(all_means_array.T, all_stds_array.T):
        plt.plot(mean, linestyle="-", marker=".")
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.4)

    plt.legend(methods.keys())
    # plt.title("Computation time per each step")
    plt.ylabel("Time in seconds")
    plt.xlabel("Number of parameters")
    plt.yscale("log")
    plt.savefig("computational_cost.pdf")


if __name__ == "__main__":
    main()
