import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def main():

    methods = [
        "Adam",
        "SGD",
        "BL89",
        "AdaHessian",
        "GGNMC",
        "AdaHesScaleLM",
        "AdaHesScale",
    ]


    for single_hidden_layer in [True, False]:
        dirName = (
            "data/ex_cost_comp_single_layer" if single_hidden_layer else "data/ex_cost_comp"
        )
        data_name = "data_single_layer_" if single_hidden_layer else "data_"


        means = np.load(f"{dirName}/{data_name}means.npy")
        stderrs = np.load(f"{dirName}/{data_name}stderrs.npy")
        n_params = np.load(f"{dirName}/{data_name}n_params.npy")

        for mean, stderr in zip(means.T, stderrs.T):
            plt.plot(n_params, mean, linestyle="-", marker=".")
            plt.fill_between(range(len(mean)), mean - stderr, mean + stderr, alpha=0.4)

        plt.legend(methods)
        # plt.title("Computation time per each step")
        plt.ylabel("Time in seconds")
        plt.xlabel("Number of parameters")
        plt.xscale("log", base=2)
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


if __name__ == "__main__":
    main()
