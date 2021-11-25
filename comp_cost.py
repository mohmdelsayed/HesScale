import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from backpack import backpack, extend
from backpack.extensions import DiagGGNExact, DiagGGNMC, DiagHessian
from torch import nn
from torchvision.models import resnet50

from hess_bench.optimizers.ada_hessian import Adahessian
from hess_bench.optimizers.exact_diag_hess import ExactHessDiagOptimizer
from hess_bench.optimizers.ggn import GGNExactOptimizer
from hess_bench.optimizers.ggn_mc import GGNMCOptimizer
from hess_bench.optimizers.hesscale import HesScaleOptimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", required=False, type=int, default="0")
    parser.add_argument("-d", "--device", required=False)
    parser.add_argument("-n", "--netmode", required=False)
    parser.add_argument("-o", "--optmode", required=False)
    args = parser.parse_args()

    device = "cpu"
    if args.device:
        device = args.device

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.set_deterministic(True)

    # Solution
    netmode = "cnn"
    if args.netmode:
        netmode = args.netmode

    if netmode == "mlp":
        ninpts = 20
        nhidden = 2048
        lnet = torch.nn.Sequential(
            nn.Linear(ninpts, nhidden, bias=False),
            nn.Tanh(),
            nn.Linear(nhidden, nhidden, bias=False),
            nn.Tanh(),
            nn.Linear(nhidden, nhidden, bias=False),
            nn.Tanh(),
            nn.Linear(nhidden, nhidden, bias=False),
            nn.Tanh(),
            nn.Linear(nhidden, nhidden, bias=False),
            nn.Tanh(),
            nn.Linear(nhidden, nhidden, bias=False),
            nn.Tanh(),
            nn.Linear(nhidden, nhidden, bias=False),
            nn.Tanh(),
            nn.Linear(nhidden, nhidden, bias=False),
            nn.Tanh(),
            nn.Linear(nhidden, nhidden, bias=False),
            nn.Tanh(),
            nn.Linear(nhidden, nhidden, bias=False),
            nn.Tanh(),
            nn.Linear(nhidden, nhidden, bias=False),
            nn.Tanh(),
            nn.Linear(nhidden, nhidden, bias=False),
            nn.Tanh(),
            nn.Linear(nhidden, nhidden, bias=False),
            nn.Tanh(),
            nn.Linear(nhidden, nhidden, bias=False),
            nn.Tanh(),
            nn.Linear(nhidden, nhidden, bias=False),
            nn.Tanh(),
            nn.Linear(nhidden, 100, bias=False),
        ).to(device)
        # print(count_parameters(lnet))

    elif netmode == "cnn":
        imgwid = 512
        nchannel = 3
        nhidden = 8
        lnet = torch.nn.Sequential(
            nn.Conv2d(nchannel, nhidden, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(nhidden, nhidden, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(nhidden, nhidden, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(nhidden, nhidden, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(nhidden, nhidden, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(nhidden, nhidden, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(nhidden * 4 * 4, 20),
        ).to(device)
    elif netmode == "resnet":
        imgwid = 512
        nchannel = 3
        lnet = resnet50()
        lnet.fc = nn.Linear(2048, 1)
        lnet = lnet.to(device)
        # print(lnet)
        # print(next(lnet.parameters()).is_cuda)
        # print(next(lnet.layer4.parameters()).is_cuda)
    else:
        print("invalid netmode")
        sys.exit(0)

    method_codes = ["adahess", "ggn", "ggnmc", "hesscale", "adam", "sgd"]
    sec_order = {
        "h": ExactHessDiagOptimizer.method,
        "ggn": GGNExactOptimizer.method,
        "ggnmc": GGNMCOptimizer.method,
        "hesscale": HesScaleOptimizer.method,
    }
    means = []
    stds = []

    for method_code in method_codes:

        if (
            method_code == "ggn"
            or method_code == "ggnmc"
            or method_code == "hesscale"
            or method_code == "h"
        ):
            extend(lnet)
            lossf = extend(nn.MSELoss())
        else:
            lossf = nn.MSELoss()

        # Experiments
        T = 10
        mses = np.zeros((1, T))
        loop_times = np.zeros(T)
        max_mem = np.zeros(T)
        mem = np.zeros(T)
        optmode = method_code
        if args.optmode:
            optmode = args.optmode
        if optmode == "sgd":
            opt = optim.SGD(lnet.parameters(), lr=1e-4)
        elif optmode == "adam":
            opt = optim.Adam(lnet.parameters())
        elif optmode == "obd":
            opt = HesScaleOptimizer(lnet.parameters(), betas=[0, 0.999])
        elif optmode == "ggn":
            opt = GGNExactOptimizer(lnet.parameters(), betas=[0, 0.999])
        elif optmode == "ggnmc":
            opt = GGNMCOptimizer(lnet.parameters(), betas=[0, 0.999])
        elif optmode == "h":
            opt = ExactHessDiagOptimizer(lnet.parameters(), betas=[0, 0.999])
        elif optmode == "adahess":
            opt = Adahessian(lnet.parameters(), betas=[0, 0.999])

        for t in range(T):
            # torch.cuda.reset_peak_memory_stats()
            prev_t = time.perf_counter()
            if netmode == "mlp":
                x = torch.randn((1, ninpts))  # .to(device)
            else:
                x = torch.randn((1, nchannel, imgwid, imgwid))  # .to(device)

            y = torch.randn((1, 20))  # .to(device)
            ypred = lnet(x)
            loss = lossf(ypred, y)
            opt.zero_grad()

            if (
                method_code == "ggn"
                or method_code == "ggnmc"
                or method_code == "obd"
                or method_code == "h"
            ):
                with backpack(sec_order[method_code]):
                    loss.backward()
            else:
                loss.backward(create_graph=True)
            opt.step()
            mses[0, t] = loss.item()
            loop_times[t] = time.perf_counter() - prev_t
            # max_mem[t] = torch.cuda.max_memory_allocated()
            # mem[t] = torch.cuda.memory_allocated()

        print("any nans: ", any([p.isnan().any().item() for p in lnet.parameters()]))
        print(mses[:, 2 * T // 3 :].mean(1))
        print("loop: ", loop_times.mean())
        means.append(loop_times.mean())
        stds.append(2.0 * loop_times.std())

    plt.bar(method_codes, means, yerr=stds / np.sqrt(50) * 2)
    plt.title("Computation time per each step")
    plt.ylabel("Time in seconds")
    plt.show()


if __name__ == "__main__":
    main()
