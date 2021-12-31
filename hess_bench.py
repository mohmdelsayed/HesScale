import os
import random
import warnings
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from _color_data import TABLEAU_COLORS, XKCD_COLORS
from hess_bench.experiment import HessExperimentComparision
from hess_bench.utils import unpack

warnings.filterwarnings("ignore")

n_plots = 4
@unpack
def run(configs, seed, lamda):
    experiment = HessExperimentComparision(configs, seed)
    # print(f"Started run with seed: {experiment.seed}")
    return experiment.train(lamda)


def main():
    configs_dir = "hess_bench/configs_"

    data_lamda1 = {}
    for configs in os.listdir(configs_dir):
        with open(f"{configs_dir}/{configs}") as file:
            configs = yaml.full_load(file)

        random.seed(54321)
        randomlist = random.sample(range(0, 99999), configs["n_seeds"])
        exp_name = configs["exp_name"]

        lamda_range = list(np.arange(-4, 6) * 0.125 + 0.5)
        for lamda in lamda_range:
            data = {}
            pool = Pool(configs["n_processes"])
            results = pool.map(run, [(configs, seed, lamda) for seed in randomlist])
            pool.close()
            pool.join()

            for result in results:
                for method in result:
                    sums = np.zeros_like(result[method][list(result[method].keys())[0]])

                    for name in result[method]:
                        sums += result[method][name]

                    if method in data:
                        data[method].append(sums)
                    else:
                        data[method] = [sums]

            cats, bar_means, bar_stds = [], [], []
            for element in data:
                bar_means.append(np.asarray(data[element]).mean(axis=1).mean(axis=0))
                bar_stds.append(np.asarray(data[element]).mean(axis=0).std())
                cats.append(element)
            if lamda == 1.0:
                data_lamda1 = data.copy()
                plt.bar(
                    cats,
                    bar_means,
                    yerr=bar_stds
                    / (
                        np.sqrt(
                            configs["n_seeds"]
                            * configs["data_generator_params"]["dataset_size"]
                        )
                        * 2
                    ),
                    color="gray",
                )
            plt.title(
                "Quality of Diagonal Hessian Approximations in {}".format(
                    configs["env_name"]
                )
            )
            plt.yscale('log')
            plt.ylabel("L1 Error")

            plt.plot(
                cats,
                bar_means,
                linewidth=0.8,
            )
        plt.legend(["λ = {}".format(lamda_i) for lamda_i in lamda_range])
        plt.show()

        # Figure 2:
        nums = {}
        for i, result in enumerate(results):
            for method in data_lamda1:
                if not method in nums:
                    nums[method] = np.zeros((len(results), n_plots))
                hess_list = []
                w = 0.0
                for name in result[method]:
                    item = sum(result[method][name]) / len(result[method][name])
                    if name[-6] == "w":
                        w += item
                    if name[-4] == "b":
                        w += item
                        hess_list.append(w)
                        w = 0.0

                nums[method][i, :] = np.asarray(hess_list)

        # del nums["|H|"]
        for method in nums:
            plt.plot(range(1,n_plots+1), nums[method].mean(axis=0))
            plt.yscale('log')
        plt.legend([method for method in nums])
        plt.title("Quality with number of layers")
        plt.ylabel("L1 Error")
        plt.xlabel("Layer Number")
        plt.xlim(left=0.0, right=n_plots+1)
        plt.ylim(bottom=0.0)
        plt.show()
        
        


        # Figure 3:
        w = 0.8  # bar width
        colors = [XKCD_COLORS[key] for key in XKCD_COLORS]
        x = range(len(cats))  # x-coordinates of your bars
        y = [np.asarray(data_lamda1[label]).mean(axis=1) for label in cats]
        fig, ax = plt.subplots()
        normalized_y = y / y[0]
        ax.bar(
            x,
            height=[np.mean(yi) / y[0].mean() for yi in y],
            # yerr=[np.std(normalized_y)/ (np.sqrt(configs["n_seeds"]) * 2) for yi in y],
            capsize=4,  # error bar cap width in points
            width=w,  # bar width
            tick_label=cats,
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
                    # ax.plot([current_x[j], prev_x[j]], [current_y[j], prev_y[j]], zorder=22, color=colors[j], alpha=0.7, linewidth=0.5)
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

        plt.xticks(x, cats)
        plt.axhline(y=1.0, color="r", alpha=0.2, linestyle="-", zorder=-12)
        plt.ylabel("Normalized L1 error")
        plt.show()


if __name__ == "__main__":
    main()
