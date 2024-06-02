import json
import matplotlib.pyplot as plt
from core.best_config import BestConfig
import os
import numpy as np
import matplotlib
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({'font.size': 12})

class SLPlotter:
    def __init__(self, best_runs_path, task_name, avg_interval=10000, what_to_plot="losses"):
        self.best_runs_path = best_runs_path
        self.avg_interval = avg_interval
        self.task_name = task_name
        self.what_to_plot = what_to_plot

    def plot_fig4(self, learners, colors):
        figsize = (7.767, 4.8)
        fig, axes = plt.subplots(2, 1)
        axes = np.array([axes]) if not isinstance(axes, np.ndarray) else axes.flatten()
        fig.set_size_inches(*figsize)

        for subdir, learner in zip(self.best_runs_path, learners):
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            times = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    configuration_list.append(data[self.what_to_plot])
                    times.append(data['times'])
                    learner_name = data["learner"]

            configuration_list = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // self.avg_interval, self.avg_interval).mean(axis=-1)
            times = np.array(times).reshape(len(seeds), len(times[0]))
            mean_list = np.array(configuration_list).mean(axis=0)
            std_list = np.array(configuration_list).std(axis=0) / np.sqrt(len(seeds))
            color = colors[learner]
            axes[0].plot(times[0], mean_list, label=learner_name, color=color)
            axes[0].fill_between(times[0], mean_list - std_list, mean_list + std_list, alpha=0.2, color=color)
            axes[1].plot(mean_list, label=learner_name, color=color)
            axes[1].fill_between(range(len(mean_list)), mean_list - std_list, mean_list + std_list, alpha=0.2, color=color)
            if self.what_to_plot == "losses":
                plt.ylim([0.0, 2.5])
                plt.ylabel("Online Loss")
            else:
                # axes[0].set_ylim([.2, .43])
                # axes[1].set_ylim([.2, .43])
                axes[0].set_ylim([.2, .6])
                axes[1].set_ylim([.2, .6])
                axes[0].set_ylabel("Test Accuracy")
                axes[1].set_ylabel("Test Accuracy")
        
        axes[0].set_xlabel(f"Time in seconds")
        axes[1].set_xlabel(f"Epochs")
        # plt.title(self.task_name)
        leg_handles, leg_labels = axes[0].get_legend_handles_labels()
        fig.legend(leg_handles, leg_labels, loc='lower center', ncol=4, bbox_to_anchor=(.5, -.1))
        fig.tight_layout()
        fig.savefig("plots/plot.pdf", bbox_inches='tight')
        # plt.clf()

    def plot(self):
        for subdir in self.best_runs_path:
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    configuration_list.append(data[self.what_to_plot])
                    learner_name = data["learner"]

            configuration_list = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // self.avg_interval, self.avg_interval).mean(axis=-1)
            mean_list = np.array(configuration_list).mean(axis=0)
            std_list = np.array(configuration_list).std(axis=0) / np.sqrt(len(seeds))
            plt.plot(mean_list, label=learner_name)
            plt.fill_between(range(len(mean_list)), mean_list - std_list, mean_list + std_list, alpha=0.2)
            if self.what_to_plot == "losses":
                plt.ylim([0.0, 2.5])
                plt.ylabel("Online Loss")
            else:
                plt.ylim([.2, .45])
                plt.ylabel("Test Accuracy")
            plt.legend()
        
        plt.xlabel(f"Epochs")
        # plt.title(self.task_name)
        plt.savefig("plots/plot.pdf", bbox_inches='tight')
        plt.clf()

if __name__ == "__main__":
    # learners = ['sgd']
    learners = ['sgd', 'adam', 'adahessian', 'adaggnmc', 'adahesscale_adamstyle', 'adahesscalegn_adamstyle']
    colors = {
        'sgd': 'tab:red',
        'adam': 'tab:pink',
        'adahessian': 'tab:brown',
        'adaggnmc': 'tab:blue',
        'adahesscale_adamstyle': 'tab:green',
        'adahesscalegn_adamstyle': 'tab:orange',
    }
    best_runs = BestConfig('sl_3c3d', 'cifar-100', 'net_cifar10_3c3d', learners, learners).get_best_run_sl(measure='accuracies_test')
    print(best_runs)
    plotter = SLPlotter(best_runs, task_name='cifar-100', avg_interval=1, what_to_plot='accuracies_test')
    plotter.plot_fig4(learners, colors)
