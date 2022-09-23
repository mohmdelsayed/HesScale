import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from utils import TABLEAU_COLORS, XKCD_COLORS

matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def main():

    dir_name = "data/ex_approx_quality"
    file_name = "data"
    normalizer = "HS"

    with open(f"{dir_name}/{file_name}.pkl", "rb") as f:
        L1_errors_per_method = pickle.load(f)[1]

    figure(L1_errors_per_method, dir_name, normalizer)


def figure(L1_errors_per_method, dir_name, normalizer):

    # Figure: Compute total L1 distance normalized by HesScale
    w = 0.8  # bar width
    colors = [XKCD_COLORS[key] for key in XKCD_COLORS]
    colors_tab = TABLEAU_COLORS

    # change how data is organized: from L1 errors per method per layer to L1 errors per method
    data = {}
    for run in L1_errors_per_method:
        for method in run:

            sums = np.zeros_like(run[method][list(run[method].keys())[0]])

            for layer in run[method]:
                sums += run[method][layer]

            if method in data:
                data[method].append(sums)
            else:
                data[method] = [sums]

    methods = data.keys()
    x = range(len(methods))  # x-coordinates of your bars
    y = [np.asarray(data[method]).mean(axis=1) for method in methods]
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,5), sharex=True, gridspec_kw={'height_ratios': [1, 4]})
    fig.subplots_adjust(hspace=0.025)  # adjust space between axes

    normalized_y = y / y[list(methods).index(normalizer)]
    ax1.bar(
        x,
        height=[np.mean(yi) / y[list(methods).index(normalizer)].mean() for yi in y],
        capsize=4,  # error bar cap width in points
        width=w,  # bar width
        tick_label=methods,
        color=colors_tab,  # face color transparent
        alpha=0.5,
    )
    ax2.bar(
        x,
        height=[np.mean(yi) / y[list(methods).index(normalizer)].mean() for yi in y],
        capsize=4,  # error bar cap width in points
        width=w,  # bar width
        tick_label=methods,
        color=colors_tab,  # face color transparent
        alpha=0.5,
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
    ax1.set_ylim(700, 1100)  # outliers only
    ax2.set_ylim(0, 35)  # most of the data

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=6,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    plt.xticks(x, methods)
    plt.axhline(y=1.0, color="r", alpha=0.2, linestyle="-", zorder=-12)
    plt.ylabel("Normalized L1 error")
    plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
    plt.savefig(f"{dir_name}/figure.pdf")


if __name__ == "__main__":
    main()
