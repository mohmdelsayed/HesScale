import time, math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from backpack import backpack, extend
from torch import nn
import warnings
import os
import matplotlib

warnings.filterwarnings("ignore")
from experiments.computational_cost.optimizers.ada_hessian import Adahessian
from experiments.computational_cost.optimizers.exact_diag_hess import (
    ExactHessDiagOptimizer,
)
from experiments.computational_cost.optimizers.ggn import GGNExactOptimizer
from experiments.computational_cost.optimizers.ggn_mc import GGNMCOptimizer
from experiments.computational_cost.optimizers.ada_hesscale import HesScaleOptimizer
from experiments.computational_cost.optimizers.bl89 import BL89Optimizer
from experiments.computational_cost.optimizers.ada_hesscale_lm import HesScaleLMOptimizer
from experiments.computational_cost.optimizers.kfac import KFACOptimizer


matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def main():

    np.random.seed(1234)
    torch.manual_seed(1234)

    methods = {
        "Adam": {"class": optim.Adam, "backpack": False},
        "SGD": {"class": optim.SGD, "backpack": False},
        "BL89": {"class": BL89Optimizer, "backpack": True},
        "AdaHessian": {"class": Adahessian, "backpack": False},
        "GGNMC": {"class": GGNMCOptimizer, "backpack": True},
        "AdaHesScaleLM": {"class": HesScaleLMOptimizer, "backpack": True},
        "AdaHesScale": {"class": HesScaleOptimizer, "backpack": True},
        # "GGN": {"class": GGNExactOptimizer, "backpack": True},
        # "KFAC": {"class": KFACOptimizer, "backpack": True},
        # "H": {"class": ExactHessDiagOptimizer, "backpack": True},
    }

    all_means = []
    all_stds = []
    T = 30
    singlelayer_exp_list = [256, 512, 1024, 2048, 4096, 8192]
    non_singlelayer_exp_list = [1, 2, 4, 8, 16, 32, 64, 128]

    single_hidden_layer = False
    n_inputs = 64
    dirName = (
        "data/ex_cost_comp_single_layer" if single_hidden_layer else "data/ex_cost_comp"
    )
    data_name = "data_single_layer_" if single_hidden_layer else "data_"

    if not os.path.exists(dirName):
        # necessary to make the script cpu intensive to prevent the use of efficiency cores (M1 chip)
        s=0
        for i in range(100000000):
            s+=1
        list_range = (
            singlelayer_exp_list if single_hidden_layer else non_singlelayer_exp_list
        )
        n_params = []
        for output in list_range:
            means = []
            stds = []
            if single_hidden_layer:
                nfeatures = 256
                layers = [
                    nn.Linear(n_inputs, nfeatures, bias=False),
                    nn.Linear(nfeatures, output, bias=False),
                ]
                net = nn.Sequential(*layers)
            else:
                hidden_units = 512
                layers = [nn.Linear(n_inputs, hidden_units, bias=False)]
                for i in range(output):
                    layers.append(nn.Linear(hidden_units, hidden_units, bias=False))
                    layers.append(nn.Tanh())
                layers.append(nn.Linear(hidden_units, 100, bias=False))
                net = nn.Sequential(*layers)
                output = 100
            print("Number of params:", sum(p.numel() for p in net.parameters()))
            n_params.append(sum(p.numel() for p in net.parameters()))

            for method_name in methods:

                opt, opt_class_backpack = get_optimizer(net, methods, method_name)

                if opt_class_backpack is not None:
                    extend(net)
                    lossf = extend(nn.MSELoss())
                else:
                    lossf = nn.MSELoss()

                # Experiments
                loop_times = np.zeros(T)

                for t in range(T):
                    x = torch.randn((1, n_inputs))
                    y = torch.randn((1, output))
                    loss = lossf(net(x), y)
                    opt.zero_grad()

                    if opt_class_backpack is not None:
                        with backpack(opt_class_backpack.method):
                            prev_t = time.perf_counter()
                            loss.backward()
                    else:
                        if type(opt) == Adahessian:
                            prev_t = time.perf_counter()
                            loss.backward(create_graph=True)
                        else:
                            prev_t = time.perf_counter()
                            loss.backward()

                    opt.step()
                    current_t = time.perf_counter()
                    loop_times[t] = current_t - prev_t
                means.append(loop_times.mean())
                stds.append(2.0 * loop_times.std() / math.sqrt(T))

            all_means.append(means)
            all_stds.append(stds)

        os.makedirs(dirName)
        all_means_array = np.asarray(all_means)
        all_stds_array = np.asarray(all_stds)

        np.save(f"{dirName}/{data_name}means.npy", np.asarray(all_means))
        np.save(f"{dirName}/{data_name}stds.npy", np.asarray(all_stds))
        np.save(f"{dirName}/{data_name}n_params.npy", np.asarray(n_params))

    else:
        all_means_array = np.load(f"{dirName}/{data_name}means.npy")
        all_stds_array = np.load(f"{dirName}/{data_name}stds.npy")
        n_params = np.load(f"{dirName}/{data_name}n_params.npy")

    for mean, std in zip(all_means_array.T, all_stds_array.T):
        plt.plot(n_params, mean, linestyle="-", marker=".")
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.4)

    plt.legend(methods.keys())
    # plt.title("Computation time per each step")
    plt.ylabel("Time in seconds")
    plt.xlabel("Number of parameters")
    plt.xscale("log", basex=2)
    low_xlim = 2 ** math.floor(math.log2(n_params[0]))
    high_xlim = 2 ** math.ceil(math.log2(n_params[-1]))
    plt.xlim([low_xlim, high_xlim])
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
