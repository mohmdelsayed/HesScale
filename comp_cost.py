import time, math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from backpack import backpack, extend
from backpack.extensions import DiagGGNExact, DiagGGNMC, DiagHessian, KFAC
from torch import nn

from hess_bench.optimizers.ada_hessian import Adahessian
from hess_bench.optimizers.exact_diag_hess import ExactHessDiagOptimizer
from hess_bench.optimizers.ggn import GGNExactOptimizer
from hess_bench.optimizers.ggn_mc import GGNMCOptimizer
from hess_bench.optimizers.hesscale import HesScaleOptimizer
from hess_bench.optimizers.kfac import KFACOptimizer


def main():
    np.random.seed(1234)
    torch.manual_seed(1234)
    method_codes = ["adahess", "ggn", "ggnmc", "hesscale", "adam", "sgd", "kfac"]
    all_means = []
    all_stds = []
    T = 5
    option_quadratic = False
    list_range = (
        [16, 32, 64, 128, 256, 512] if option_quadratic else [2, 4, 8, 16, 32, 64]
    )

    for output in list_range:
        if option_quadratic:
            ninpts = 100
            layers = [
                nn.Linear(1, ninpts, bias=False),
                nn.Linear(ninpts, output, bias=False),
            ]
            lnet = nn.Sequential(*layers)
        else:
            ninpts = 1
            nhidden = 256
            layers = [nn.Linear(ninpts, nhidden, bias=False)]
            for i in range(output):
                layers.append(nn.Linear(nhidden, nhidden, bias=False))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(nhidden, 2, bias=False))
            lnet = nn.Sequential(*layers)
            output = 1
        print("Number of params:", sum(p.numel() for p in lnet.parameters()))

        sec_order = {
            "h": ExactHessDiagOptimizer.method,
            "ggn": GGNExactOptimizer.method,
            "ggnmc": GGNMCOptimizer.method,
            "hesscale": HesScaleOptimizer.method,
            "kfac": KFACOptimizer.method,
        }

        means = []
        stds = []

        for method_code in method_codes:
            if method_code == "sgd":
                opt = optim.SGD(lnet.parameters(), lr=1e-4)
            elif method_code == "adam":
                opt = optim.Adam(lnet.parameters(), betas=[0, 0.999])
            elif method_code == "obd":
                opt = HesScaleOptimizer(lnet.parameters(), betas=[0, 0.999])
            elif method_code == "ggn":
                opt = GGNExactOptimizer(lnet.parameters(), betas=[0, 0.999])
            elif method_code == "ggnmc":
                opt = GGNMCOptimizer(lnet.parameters(), betas=[0, 0.999])
            elif method_code == "h":
                opt = ExactHessDiagOptimizer(lnet.parameters(), betas=[0, 0.999])
            elif method_code == "adahess":
                opt = Adahessian(lnet.parameters(), betas=[0, 0.999])
            elif method_code == "kfac":
                opt = KFACOptimizer(lnet.parameters(), betas=[0, 0.999])

            if method_code in ["ggn", "ggnmc", "hesscale", "h", "kfac"]:
                extend(lnet)
                lossf = extend(nn.MSELoss())
            else:
                lossf = nn.MSELoss()

            # Experiments
            mses = np.zeros((1, T))
            loop_times = np.zeros(T)

            for t in range(T):
                prev_t = time.perf_counter()
                x = torch.randn((1, 1))
                y = torch.randn((1, output))
                loss = lossf(lnet(x), y)
                opt.zero_grad()

                if method_code in ["ggn", "ggnmc", "hesscale", "h", "kfac"]:
                    with backpack(sec_order[method_code]):
                        loss.backward()
                else:
                    loss.backward(create_graph=True)

                opt.step()
                mses[0, t] = loss.item()
                loop_times[t] = time.perf_counter() - prev_t

            means.append(loop_times.mean())
            stds.append(2.0 * loop_times.std() / math.sqrt(T))

        all_means.append(means)
        all_stds.append(stds)

    all_means_array = np.asarray(all_means)
    all_stds_array = np.asarray(all_stds)

    for mean, std in zip(all_means_array.T, all_stds_array.T):
        plt.plot(mean)
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.4)

    plt.legend(method_codes)
    plt.title("Computation time per each step")
    plt.ylabel("Time in seconds")
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
