import os
import random
import warnings
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
import pickle
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

    dir_name = "data/ex_approx_quality"
    configs_dir = "experiments/approximation_quality/configs"
    file_name = "data_lambdas"
    lamda_range = [1.0]  # list(np.arange(-4, 6) * 0.125 + 0.5)
    normalizer = "HesScale"

    data_lambdas = {}
    with open(f"{configs_dir}/configs.yaml") as file:
        configs = yaml.full_load(file)

    n_seeds = configs["n_seeds"]
    random.seed(54321)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        randomlist = random.sample(range(0, 99999), n_seeds)
        for lamda in lamda_range:
            pool = Pool(configs["n_processes"])
            results = pool.map(run, [(configs, seed, lamda) for seed in randomlist])
            pool.close()
            pool.join()
            data_lambdas[lamda] = results

        with open(f"{dir_name}/{file_name}.pkl", "wb") as f:
            pickle.dump(data_lambdas, f)
    else:
        with open(f"{dir_name}/{file_name}.pkl", "rb") as f:
            data_lambdas = pickle.load(f)

    figure_1(lamda_range, data_lambdas, n_seeds, dir_name)
    figure_2(data_lambdas[1], dir_name)
    figure_3(data_lambdas[1], dir_name, normalizer)


def figure_1(lamda_range, data_lambdas, n_seeds, dir_name):
    # Figure 1: plot the summed errors over the number of samples per each method
    for lamda in lamda_range:
        data = {}
        for result in data_lambdas[lamda]:
            for method in result:
                sums = np.zeros_like(result[method][list(result[method].keys())[0]])

                for name in result[method]:
                    sums += result[method][name]

                if method in data:
                    data[method].append(sums)
                else:
                    data[method] = [sums]

        # per each method, compute bar means and std errors
        bar_means, bar_stds = [], []
        methods = data.keys()

        for method in data:
            bar_means.append(np.asarray(data[method]).mean(axis=1).mean(axis=0))
            bar_stds.append(np.asarray(data[method]).mean(axis=1).std())

        # save data for lambda=1 in a seperate variable, then plot bars for lambda=1
        if lamda == 1.0:
            plt.bar(
                methods,
                bar_means,
                yerr=bar_stds / (np.sqrt(n_seeds) * 2),
                color="gray",
            )

        # plot connected points for each lambda
        plt.plot(
            methods,
            bar_means,
            linewidth=0.8,
        )
        plt.title("Quality of Diagonal Hessian Approximations")
        plt.yscale("log")
        plt.ylabel("L1 Error")

    plt.legend(["λ = {}".format(lamda_i) for lamda_i in lamda_range])
    plt.savefig(f"{dir_name}/plot1.pdf")
    plt.clf()


def figure_2(data_lamda1, dir_name):
    # Figure 2: compute sum errors over the number of samples per each method per each layer
    errors_per_method_per_layer = {}
    methods = data_lamda1[0].keys()
    for i, result in enumerate(data_lamda1):
        for method in methods:
            if not method in errors_per_method_per_layer:
                errors_per_method_per_layer[method] = np.zeros(
                    (len(data_lamda1), len(result[method]) // 2)
                )

            erros_per_layer = []
            error_sum = 0.0
            for name in result[method]:
                item = sum(result[method][name]) / len(result[method][name])
                if "w" in name:
                    error_sum += item
                if "b" in name:
                    error_sum += item
                    erros_per_layer.append(error_sum)
                    error_sum = 0.0

            errors_per_method_per_layer[method][i, :] = np.asarray(erros_per_layer)

    for method in errors_per_method_per_layer:
        plt.plot(
            range(1, len(result[method]) // 2 + 1),
            errors_per_method_per_layer[method].mean(axis=0),
        )
        plt.yscale("symlog", linthreshy=1e-1)
    plt.legend([method for method in errors_per_method_per_layer])
    # plt.title("Quality with number of layers")
    plt.xticks(range(1, len(result[method]) // 2 + 1))
    plt.ylabel("L1 Error")
    plt.xlabel("Layer Number")
    plt.ylim(bottom=0.0)
    plt.savefig(f"{dir_name}/plot2.pdf")
    plt.clf()


def figure_3(data_lambda1, dir_name, normalizer):

    # Figure 3: Compute total L1 distance normalized by HesScale
    w = 0.8  # bar width
    colors = [XKCD_COLORS[key] for key in XKCD_COLORS]
    data = {}
    for result in data_lambda1:
        for method in result:
            sums = np.zeros_like(result[method][list(result[method].keys())[0]])

            for name in result[method]:
                sums += result[method][name]

            if method in data:
                data[method].append(sums)
            else:
                data[method] = [sums]

    methods = data.keys()
    x = range(len(methods))  # x-coordinates of your bars
    y = [np.asarray(data[method]).mean(axis=1) for method in methods]
    fig, ax = plt.subplots()

    normalized_y = y / y[list(methods).index(normalizer)]
    ax.bar(
        x,
        height=[np.mean(yi) / y[list(methods).index(normalizer)].mean() for yi in y],
        capsize=4,  # error bar cap width in points
        width=w,  # bar width
        tick_label=methods,
        color=colors,  # face color transparent
        alpha=0.3,
    )

    # plot scattered dots around the bar of each method
    rand_number = np.random.uniform(low=-w / 2, high=w / 2, size=normalized_y.shape[1])
    repeated_x = np.repeat(np.asarray(x).reshape(-1, 1), normalized_y.shape[1], axis=1)
    x_matrix = repeated_x + rand_number
    y_matrix = normalized_y

    for x_method, y_method in zip(x_matrix, y_matrix):
        ax.scatter(
            x_method,
            y_method,
            zorder=54,
            color=colors[: len(x_method)],
            alpha=1.0,
            s=1.4,
        )

    plt.xticks(x, methods)
    plt.axhline(y=1.0, color="r", alpha=0.2, linestyle="-", zorder=-12)
    plt.ylabel("Normalized L1 error")
    plt.savefig(f"{dir_name}/plot3.pdf")
    plt.clf()


if __name__ == "__main__":
    main()
