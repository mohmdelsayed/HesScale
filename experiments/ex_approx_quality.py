import os
import random
import warnings
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from utils import TABLEAU_COLORS, XKCD_COLORS
from experiments.approximation_quality.experiment import HessQualityExperiment
from experiments.approximation_quality.utils import unpack


matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
warnings.filterwarnings("ignore")

@unpack
def run(configs, seed, lamda):
    experiment = HessQualityExperiment(configs, seed)
    # print(f"Started run with seed: {experiment.seed}")
    return experiment.train(lamda)


def main():

    configs_dir = "experiments/approximation_quality"
    data_lamda1 = {}
    with open(f"{configs_dir}/configs.yaml") as file:
        configs = yaml.full_load(file)

    random.seed(54321)
    randomlist = random.sample(range(0, 99999), configs["n_seeds"])
    lamda_range = [1.0]  # list(np.arange(-4, 6) * 0.125 + 0.5)
    
    for lamda in lamda_range:
        data_lamdas = {}
        pool = Pool(configs["n_processes"])
        results = pool.map(run, [(configs, seed, lamda) for seed in randomlist])
        pool.close()
        pool.join()

        for result in results:
            for method in result:
                sums = np.zeros_like(result[method][list(result[method].keys())[0]])

                for name in result[method]:
                    sums += result[method][name]

                if method in data_lamdas:
                    data_lamdas[method].append(sums)
                else:
                    data_lamdas[method] = [sums]


        bar_means, bar_stds = [], []
        methods = data_lamdas.keys()
        for method in data_lamdas:
            bar_means.append(np.asarray(data_lamdas[method]).mean(axis=1).mean(axis=0))
            bar_stds.append(np.asarray(data_lamdas[method]).mean(axis=1).std())

        if lamda == 1.0:
            data_lamda1 = data_lamdas.copy()
            plt.bar(
                methods,
                bar_means,
                yerr=bar_stds
                / (
                    np.sqrt(
                        configs["n_seeds"]
                    )
                    * 2
                ),
                color="gray",
            )

        plt.plot(
            methods,
            bar_means,
            linewidth=0.8,
        )
        plt.title("Quality of Diagonal Hessian Approximations")
        plt.yscale("log")
        plt.ylabel("L1 Error")

    plt.legend(["λ = {}".format(lamda_i) for lamda_i in lamda_range])
    plt.savefig("plot1.pdf")
    plt.clf()


    # Figure 2:
    errors = {}
    for i, result in enumerate(results):
        for method in methods:
            if not method in errors:
                errors[method] = np.zeros((len(results), len(result[method])//2))
            
            hess_list = []
            error_sum = 0.0
            for name in result[method]:
                item = sum(result[method][name]) / len(result[method][name])
                if 'w' in name:
                    error_sum += item
                if 'b' in name:
                    error_sum += item
                    hess_list.append(error_sum)
                    error_sum = 0.0

            errors[method][i, :] = np.asarray(hess_list)

    for method in errors:
        plt.plot(range(1, len(result[method])//2 + 1), errors[method].mean(axis=0))
        plt.yscale("symlog", linthreshy=1e-1)
    plt.legend([method for method in errors])
    # plt.title("Quality with number of layers")
    plt.xticks(range(1, len(result[method])//2 + 1))
    plt.ylabel("L1 Error")
    plt.xlabel("Layer Number")
    plt.ylim(bottom=0.0)
    plt.savefig("plot2.pdf")
    plt.clf()

    # Figure 3:
    w = 0.8  # bar width
    colors = [XKCD_COLORS[key] for key in XKCD_COLORS]
    x = range(len(methods))  # x-coordinates of your bars
    y = [np.asarray(data_lamda1[method]).mean(axis=1) for method in methods]
    fig, ax = plt.subplots()
    normalized_y = y / y[0]
    ax.bar(
        x,
        height=[np.mean(yi) / y[0].mean() for yi in y],
        # yerr=[np.std(normalized_y)/ (np.sqrt(configs["n_seeds"]) * 2) for yi in y],
        capsize=4,  # error bar cap width in points
        width=w,  # bar width
        tick_label=methods,
        color=colors,  # face color transparent
        # edgecolor=colors,
        alpha=0.3,
        # ecolor="blue",    # error bar colors; setting this raises an error for whatever reason.
        # lw=2,
    )

    rand_number = np.random.uniform(
        low=-w / 2, high=w / 2, size=normalized_y.shape[1]
    )
    for i in range(len(x)):
        current_x = x[i] + rand_number
        current_y = normalized_y[i]  # normalize by HesScale
        if "prev_x" in locals():
            for j in range(current_x.shape[0]):
                ax.scatter(
                    [current_x[j], prev_x[j]],
                    [current_y[j], prev_y[j]],
                    zorder=54,
                    color=colors[j],
                    alpha=1.0,
                    s=1.4,
                )
        prev_x = current_x
        prev_y = current_y

    plt.xticks(x, methods)
    plt.axhline(y=1.0, color="r", alpha=0.2, linestyle="-", zorder=-12)
    plt.ylabel("Normalized L1 error")
    plt.savefig("plot3.pdf")
    plt.clf()


if __name__ == "__main__":
    main()
