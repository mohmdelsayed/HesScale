import json, os
import matplotlib.pyplot as plt
from core.best_run import BestRun
from core.utils import XKCD_COLORS
import numpy as np
import matplotlib

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False

class ApproxQualityPlotter:
    def __init__(self, best_runs_path, task_name):
        self.best_runs_path = best_runs_path
        self.task_name = task_name
        self.normalizer = "HS"
        self.colors = [
            'tab:green',
            'tab:purple',
            'tab:brown',

            "#CD5C5C", #indianred

            "tab:gray",
            "tab:orange",
            "tab:blue",

            "skyblue",
            
            "tab:olive",
            
            "#DEB887", #burlywood
            "#FFA07A", #lightsalmon
            "#607c8e", #blue grey
            "#556B2F", #darkolivegreen
        ]

    def plot_l1_error(self):

        for subdir in self.best_runs_path:
            seeds = os.listdir(f'{subdir}')
            l1s_errors_before = []
            l1s_errors_after = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    l1s_errors_before.append(data["l1_errors_before"])
                    l1s_errors_after.append(data["l1_errors_after"])

        self.layerwise_error(l1s_errors_before, "before", self.colors)
        self.layerwise_error(l1s_errors_after, "after", self.colors)
        
        self.overall_error(l1s_errors_before, "before", self.colors)
        self.overall_error(l1s_errors_after, "after", self.colors)

    def layerwise_error(self, data_lamda1, file_name, colors):
        errors_per_method_per_layer = {}
        methods = data_lamda1[0].keys()
        for i, result in enumerate(data_lamda1):
            for method in methods:
                # if method == 'H':
                #     continue
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

        for method, color in zip(errors_per_method_per_layer, colors):
            plt.plot(
                range(1, len(result[method]) // 2 + 1),
                errors_per_method_per_layer[method].mean(axis=0),
                linestyle="-", marker=".",
                color=color,
            )
            plt.yscale("symlog", linthreshy=1e-1)
        plt.legend([method for method in errors_per_method_per_layer])
        # plt.title("Quality with number of layers")
        plt.xticks(range(1, len(result[method]) // 2 + 1))
        plt.ylim(bottom=0.0)
        plt.ylabel("L1 Error", fontsize=20)
        plt.xlabel("Layer Number", fontsize=20)
        plt.savefig(f"layerwise_error_{file_name}.pdf", bbox_inches='tight')
        plt.clf()

    def overall_error(self, data_lambda1, file_name, colors_tab):
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
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,5), sharex=True, gridspec_kw={'height_ratios': [1, 4]})
        fig.subplots_adjust(hspace=0.025)  # adjust space between axes

        normalized_y = y / y[list(methods).index(self.normalizer)]
        ax1.bar(
            x,
            height=[np.mean(yi) / y[list(methods).index(self.normalizer)].mean() for yi in y],
            capsize=4,  # error bar cap width in points
            width=w,  # bar width
            tick_label=methods,
            color=colors_tab,  # face color transparent
            alpha=0.75,
        )
        ax2.bar(
            x,
            height=[np.mean(yi) / y[list(methods).index(self.normalizer)].mean() for yi in y],
            capsize=4,  # error bar cap width in points
            width=w,  # bar width
            tick_label=methods,
            color=colors_tab,  # face color transparent
            alpha=0.75,
        )

        # plot scattered dots around the bar of each method
        rand_number = np.random.uniform(low=-w / 2, high=w / 2, size=normalized_y.shape[1])
        repeated_x = np.repeat(np.asarray(x).reshape(-1, 1), normalized_y.shape[1], axis=1)
        x_matrix = repeated_x + rand_number
        y_matrix = normalized_y

        for x_method, y_method in zip(x_matrix, y_matrix):
            ax1.scatter(
                x_method,
                y_method,
                zorder=54,
                color=colors[: len(x_method)],
                alpha=1.0,
                s=1.4,
            )

            ax2.scatter(
                x_method,
                y_method,
                zorder=54,
                color=colors[: len(x_method)],
                alpha=1.0,
                s=1.4,
            )

        # zoom-in / limit the view to different portions of the data
        ax1.set_ylim(500, 1200)  # outliers only
        ax2.set_ylim(0, 42)  # most of the data

        ax2.tick_params(axis="x",direction="in", pad=-70)
        # hide the spines between ax and ax2
        ax1.spines.bottom.set_visible(False)
        ax1.tick_params(bottom=False, top=False)

        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=6,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

        plt.xticks(x, methods)
        plt.axhline(y=1.0, color="r", alpha=0.2, linestyle="-", zorder=-12)
        plt.ylabel("Normalized L1 error", fontsize=18)
        plt.xlabel("Methods", fontsize=18)
        plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
        plt.savefig(f"approx_quality_{file_name}.pdf", bbox_inches='tight')


if __name__ == "__main__":
    best_runs = BestRun("exp6/stationary_mnist", "area", "fcn_relu_small", ["adam"]).get_best_run(measure="accuracies")
    print(best_runs)
    plotter = ApproxQualityPlotter(best_runs, task_name="Approximation Quality of Hessian Diagonals")
    plotter.plot_l1_error()
