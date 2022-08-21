import time, math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from backpack import backpack, extend
from torch import nn
import warnings
import os

warnings.filterwarnings("ignore")
from experiments.computational_cost.optimizers.ada_hessian import Adahessian
from experiments.computational_cost.optimizers.exact_diag_hess import (
    ExactHessDiagOptimizer,
)
from experiments.computational_cost.optimizers.ggn import GGNExactOptimizer
from experiments.computational_cost.optimizers.ggn_mc import GGNMCOptimizer
from experiments.computational_cost.optimizers.hesscale import HesScaleOptimizer
from experiments.computational_cost.optimizers.kfac import KFACOptimizer


def main():

    np.random.seed(1234)
    torch.manual_seed(1234)

    methods = {
        "HesScale": {"class": HesScaleOptimizer, "backpack": True},
        "AdaHessian": {"class": Adahessian, "backpack": False},
        "GGN": {"class": GGNExactOptimizer, "backpack": True},
        "GGNMC/LM-HesScale": {"class": GGNMCOptimizer, "backpack": True},
        "Adam": {"class": optim.Adam, "backpack": False},
        "SGD": {"class": optim.SGD, "backpack": False},
        "KFAC": {"class": KFACOptimizer, "backpack": True},
        "H": {"class": ExactHessDiagOptimizer, "backpack": True},
    }

    all_means = []
    all_stds = []
    T = 1
    singlelayer_exp_list = [512, 1024, 2048]  # , 4096, 8192]
    non_singlelayer_exp_list = [2, 4, 8, 16]  # , 32, 64, 128,]# 256, 512, 1024]

    single_hidden_layer = False
    dirName = (
        "data/ex_cost_comp_single_layer" if single_hidden_layer else "data/ex_cost_comp"
    )
    data_name = "data_single_layer_" if single_hidden_layer else "data_"

    if not os.path.exists(dirName):
        list_range = (
            singlelayer_exp_list if single_hidden_layer else non_singlelayer_exp_list
        )

        for output in list_range:
            means = []
            stds = []
            if single_hidden_layer:
                nfeatures = 512
                layers = [
                    nn.Linear(1, nfeatures, bias=False),
                    nn.Linear(nfeatures, output, bias=False),
                ]
                net = nn.Sequential(*layers)
            else:
                ninpts = 1
                hidden_units = 512
                layers = [nn.Linear(ninpts, hidden_units, bias=False)]
                for i in range(output):
                    layers.append(nn.Linear(hidden_units, hidden_units, bias=False))
                    layers.append(nn.Tanh())
                layers.append(nn.Linear(hidden_units, 100, bias=False))
                net = nn.Sequential(*layers)
                output = 100
            # print("Number of params:", sum(p.numel() for p in net.parameters()))

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

                    if opt_class_backpack is not None:
                        with backpack(opt_class_backpack.method):
                            prev_t = time.perf_counter()
                            loss.backward()
                    else:
                        prev_t = time.perf_counter()
                        loss.backward(create_graph=True)

                    opt.step()
                    current_t = time.perf_counter()

                    loop_times[t] = current_t - prev_t
                    mses[0, t] = loss.item()

                means.append(loop_times.mean())
                stds.append(2.0 * loop_times.std() / math.sqrt(T))

            all_means.append(means)
            all_stds.append(stds)

        os.makedirs(dirName)
        all_means_array = np.asarray(all_means)
        all_stds_array = np.asarray(all_stds)
        np.save(f"{dirName}/{data_name}means.npy", np.asarray(all_means))
        np.save(f"{dirName}/{data_name}stds.npy", np.asarray(all_stds))
    else:
        all_means_array = np.load(f"{dirName}/{data_name}means.npy")
        all_stds_array = np.load(f"{dirName}/{data_name}stds.npy")

    for mean, std in zip(all_means_array.T, all_stds_array.T):
        plt.plot(mean, linestyle="-", marker=".")
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.4)

    plt.legend(methods.keys())
    # plt.title("Computation time per each step")
    plt.ylabel("Time in seconds")
    plt.xlabel("Number of parameters")
    plt.yscale("log")
    file_name = (
        f"{dirName}/computational_cost_single_layer.pdf"
        if single_hidden_layer
        else f"{dirName}/computational_cost.pdf"
    )
    plt.savefig(file_name)


def get_optimizer(lnet, methods, method_name):
    opt = methods[method_name]["class"](lnet.parameters(), lr=1e-4)
    if methods[method_name]["backpack"]:
        opt_class_backpack = methods[method_name]["class"]
    else:
        opt_class_backpack = None
    return opt, opt_class_backpack


if __name__ == "__main__":
    main()
