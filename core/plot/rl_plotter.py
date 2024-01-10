import argparse
import json
import matplotlib.pyplot as plt
import core.best_config
import os
import numpy as np
import matplotlib
from pathlib import Path
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({'font.size': 12})

def bin_episodes(t, G, bin_wid, interval=None):
    t_tails, G_means, G_stds, G_stderrs, t_bins, G_bins = [], [], [], [], [], []
    if interval is not None:
        interval_idx = np.logical_and((t > interval[0]), (t <= interval[1]))
        t, G = t[interval_idx], G[interval_idx]
    for i in range(bin_wid, t[-1] + bin_wid, bin_wid):
        bin_idx = np.logical_and((t > i - bin_wid), (t <= i))
        if not np.any(bin_idx):
            continue

        t_bin = t[bin_idx]
        G_bin = G[bin_idx]
        t_tails.append(i)
        G_means.append(np.mean(G_bin))
        G_stds.append(np.std(G_bin))
        G_stderrs.append(np.std(G_bin) / np.sqrt(G_bin.shape[0]))
        t_bins.append(t_bin)
        G_bins.append(G_bin)

    return np.array(t_tails), np.array(G_means), np.array(G_stds), np.array(G_stderrs), t_bins, G_bins

class RLPlotter:
    def __init__(self, best_runs_path, exp_name, task_name, avg_interval=10000, what_to_plot="losses", plot_id='0'):
        self.best_runs_path = best_runs_path
        self.exp_name = exp_name
        self.avg_interval = avg_interval
        self.task_name = task_name
        self.what_to_plot = what_to_plot
        self.plot_id = plot_id

    def plot(self):
        for subdir in self.best_runs_path:
            seeds = os.listdir(f'{subdir}')
            ts_list = []
            configuration_list = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    n_bins = 100
                    bin_wid = data['n_samples'] // n_bins
                    ts, rets, _, _, _, _ = bin_episodes(np.array(data['ts']), np.array(data[self.what_to_plot]), bin_wid)
                    ts_list.append(ts[:n_bins])
                    configuration_list.append(rets[:n_bins])
                    learner_name = data["learner"]
                    optim = data["optimizer"]

            ts = ts_list[0]
            n_seeds = len(seeds)
            configuration_list = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // self.avg_interval, self.avg_interval).mean(axis=-1)
            mean_list = np.array(configuration_list).mean(axis=0)
            std_list = np.array(configuration_list).std(axis=0) / np.sqrt(len(seeds))
            plt.plot(ts, mean_list, label=f'{optim}_{Path(subdir).name}')
            plt.fill_between(ts, mean_list - std_list, mean_list + std_list, alpha=0.2)
            if self.what_to_plot == "losses":
                plt.ylim([0.0, 2.5])
                plt.ylabel("Online Loss")
            elif self.what_to_plot == 'returns':
                # plt.ylim([-700.0, 1750.0])
                plt.ylim([-700.0, 12000.0])
                plt.gca().set_ylabel(f'Return\naveraged over\n{n_seeds} runs', labelpad=50, verticalalignment='center').set_rotation(0)
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            else:
                plt.ylim([0.0, 1.0])
                plt.ylabel("Online Accuracy")
            plt.legend()
        
        plt.xlabel(f"time step")
        plt.title(f'{self.task_name} - A2C')
        plt_pth = Path(f'plots/{self.exp_name}/{self.plot_id}.pdf')
        plt_pth.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plt_pth, bbox_inches='tight')
        plt.clf()

def make_plots(task_name='Ant', optim_id=0):
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=int, required=False, default=0)
    args = parser.parse_args()

    what_to_plot = "returns"

    optims_list = [
        ['sgd', 'adam', 'adam_scaled', 'adam_scaled_sqrt',],
        ['sgd', 'adam', 'adahesscalegn', 'adahesscalegn_sqrt', 'adahesscalegn_adamstyle',],
        ['sgd', 'adam', 'adahesscale', 'adahesscale_sqrt', 'adahesscale_adamstyle',],
        ['sgd', 'adam_scaled', 'adahesscalegn_scaled', 'adahesscalegn_sqrt_scaled', 'adahesscalegn_adamstyle_scaled',],
        ['sgd', 'adam_scaled', 'adahesscale_scaled', 'adahesscale_sqrt_scaled', 'adahesscale_adamstyle_scaled',],
    ]
    optims = optims_list[optim_id]
    learners = ['a2c' for _ in optims]
    plot_id = f'{task_name}_{optim_id}'

    exp_name = f"exp4_Ant5"
    best_runs = core.best_config.BestConfig(exp_name, task_name, "fcn_tanh_small", learners, optims).get_best_run(measure=what_to_plot)
    print(plot_id, best_runs)
    # best_runs[1] = best_runs[1].replace('0.0003', '0.0001')
    plotter = RLPlotter(best_runs, exp_name, task_name=task_name, avg_interval=1, what_to_plot=what_to_plot, plot_id=plot_id)
    plotter.plot()

def main():
    # for task in ['Ant', 'Walker2d', 'HalfCheetah', 'Hopper', 'InvertedDoublePendulum']:
    for task in ['InvertedDoublePendulum']:
        for i in range(5):
            make_plots(task, i)

if __name__ == "__main__":
    main()
