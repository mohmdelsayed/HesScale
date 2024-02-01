import argparse
import json
import copy
import matplotlib.pyplot as plt
from numpy.ma import alltrue
import core.best_config
import os
import re
import itertools
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

class Cache:
    def __init__(self, path) -> None:
        self.path = path
        self.data = {}
        self.updated = False
        if path.exists():
            with open(path) as f:
                self.data = json.load(f)

    def has(self, key):
        return key in self.data.keys()

    def get(self, key):
        return self.data.get(key, None)

    def set(self, key, data):
        self.data[key] = data
        self.updated = True

    def __del__(self):
        if self.updated:
            # make sure data is clean
            json.loads(json.dumps(self.data))
            with open(self.path, 'wt') as f:
                json.dump(self.data, f)
            print('Cache updated')

class RLPlotter:
    def __init__(self, best_runs_path, exp_name, task_name, avg_interval=10000, what_to_plot="losses", ylim=[-700.0, 4000.0], plot_id='0'):
        self.best_runs_path = best_runs_path
        self.exp_name = exp_name
        self.avg_interval = avg_interval
        self.task_name = task_name
        self.what_to_plot = what_to_plot
        self.plot_id = plot_id
        self.ylim = ylim

    def plot(self, cache):
        for subdir in self.best_runs_path:
            seeds = os.listdir(f'{subdir}')
            ts_list = []
            configuration_list = []
            diverged = False
            for seed in seeds:
                key = f'{subdir}/{seed}'.split('experiments/rl/')[1]
                data = cache.get(key)
                if data is None:
                    with open(f"{subdir}/{seed}") as json_file:
                        data = json.load(json_file)
                if data.get('diverged', False):
                    diverged = True
                    break
                n_bins = 100
                bin_wid = data['n_samples'] // n_bins
                ts, rets, _, _, _, _ = bin_episodes(np.array(data['ts']), np.array(data[self.what_to_plot]), bin_wid)
                if not cache.has(key):
                    data_copy = copy.deepcopy(data)
                    data_copy.update({'ts': ts.tolist(), self.what_to_plot: rets.tolist()})
                    cache.set(key, data_copy)
                ts_list.append(ts[:n_bins])
                configuration_list.append(rets[:n_bins])
                learner_name = data["learner"]
                optim = data["optimizer"]

            if diverged:
                print(f'Warning: diverged -- {subdir}')
                continue

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
                plt.ylim(self.ylim)
                # plt.ylim([-700.0, 4000.0])
                plt.gca().set_ylabel(f'Return\naveraged over\n{n_seeds} runs', labelpad=50, verticalalignment='center').set_rotation(0)
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            else:
                plt.ylim([0.0, 1.0])
                plt.ylabel("Online Accuracy")
            plt.legend()
        
        plt.xlabel(f"time step")
        algo = 'A2C' if 'a2c' in self.exp_name else 'PPO'
        plt.title(f'{self.task_name} - {algo}')
        plt_pth = Path(f'plots/{self.exp_name}/{self.plot_id}.pdf')
        # plt_pth = Path(f'plots/{self.exp_name}_lr1/{self.plot_id}.pdf')
        plt_pth.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plt_pth, bbox_inches='tight')
        plt.clf()

def make_plots(cache, task_name='Ant', optim_id=0):
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=int, required=False, default=0)
    args = parser.parse_args()

    what_to_plot = "returns"

    # # A2C
    # exp_name = 'rl_a2c_5'
    # learner = 'a2c'
    # network = 'fcn_tanh_small'

    # PPO
    exp_name = 'rl_ppo_4'
    learner = 'ppo'
    network = 'var_net'
    # filter_key = None #'delta_1e-08'
    filter_key = 'lr_0.0001'

    # optims_list = [
    #     ['sgd', 'adam', 'adam_scaled'],
    # ]

    optims_list = [
        # ['sgd', 'adam', 'adam_scaled', 'adam_scaled_sqrt',],
        # ['sgd', 'adam', 'adahesscalegn', 'adahesscalegn_sqrt', 'adahesscalegn_adamstyle',],
        # ['sgd', 'adam', 'adahesscale', 'adahesscale_sqrt', 'adahesscale_adamstyle',],
        # ['sgd', 'adam_scaled', 'adahesscalegn_scaled', 'adahesscalegn_sqrt_scaled', 'adahesscalegn_adamstyle_scaled',],
        # ['sgd', 'adam_scaled', 'adahesscale_scaled', 'adahesscale_sqrt_scaled', 'adahesscale_adamstyle_scaled',],
        # ['sgd', 'adam', 'adahessian',],
        # ['sgd_scaled', 'sgd_scaled_sqrt', 'adam_scaled', 'adam_scaled_sqrt', 'adahessian_scaled',],

        ['adam', 'adam_scaled', 'adahesscale_adamstyle', 'adahesscale_adamstyle_scaled'],
    ]

#     optims_list = [
#         ['adam_scaled', 'adam_scaled_sqrt', 'adahessian_scaled',],
#         ['adahesscalegn_scaled', 'adahesscalegn_sqrt_scaled', 'adahesscalegn_adamstyle_scaled',],
#         ['adahesscale_scaled', 'adahesscale_sqrt_scaled', 'adahesscale_adamstyle_scaled',],
#     ]

    if 'ppo' in exp_name:
        task_name += '-v2'

    ylim_high = 8000. if 'ppo' in exp_name else 4000.
    ylim = [-700., ylim_high] if task_name != 'InvertedDoublePendulum' else [-700., 12000.]

    optims = optims_list[optim_id]
    learners = [learner for _ in optims]
    plot_id = f'{task_name}_{optim_id}_lr_0.0001'

    best_runs = core.best_config.BestConfig(exp_name, task_name, network, learners, optims).get_best_run(cache, measure=what_to_plot, filter_key=filter_key)
    print(plot_id, best_runs)
    # lr1_runs = [re.sub(r'lr_.*', 'lr_1.0', run) for run in best_runs]
    # best_runs = list(itertools.chain(*[[r1, r2] for r1, r2 in zip(best_runs, lr1_runs)]))
    plotter = RLPlotter(best_runs, exp_name, task_name=task_name, avg_interval=1, what_to_plot=what_to_plot, ylim=ylim, plot_id=plot_id)
    plotter.plot(cache)

def plot_learning_curves(cache):
    # for task in ['Ant', 'Walker2d', 'HalfCheetah', 'Hopper', 'InvertedDoublePendulum']:
    for task in ['Ant']:
        for i in range(1):
            # if i in [1, 2]:
            #     continue
            make_plots(cache, task, i)

def plot_line(ax, xs, seed_ys, label='', color='C0', linestyle='-'):
    # ys dimension should be (nseeds, len(xs))
    ys = seed_ys.mean(axis=0)
    inf_idx = np.isinf(ys)
    non_inf_idx = np.logical_not(inf_idx)
    ys[inf_idx] = 1e15 * np.sign(ys[inf_idx])
    y_errs = np.zeros_like(ys)
    y_errs[non_inf_idx] = (seed_ys[:, non_inf_idx].std(axis=0) / np.sqrt(seed_ys[:, non_inf_idx].shape[0]))
    ax.plot(xs, ys, color=color, label=label, linestyle=linestyle)
    start = None
    for i in range(ys.shape[0]):
        # if only one non_inf in the middle of other infs: maybe just show an error bar
        if non_inf_idx[i] and start is None:
            start = i
        elif (inf_idx[i] or i == ys.shape[0] - 1) and start is not None:
            if i == ys.shape[0] - 1:
                i += 1
            ax.fill_between(xs[start:i], ys[start:i] - y_errs[start:i], ys[start:i] + y_errs[start:i], color=color, alpha=0.3)
            start = None

def get_sensitivity(cache, exp_name, learner, network, optim, task, seeds, filter_key=None):
    path = f"logs/{exp_name}/{task}/{learner}/{optim}/{network}/"
    all_subdirectories = [f.path for f in os.scandir(path) if f.is_dir()]
    if filter_key is not None:
        all_subdirectories = list(filter(lambda k: filter_key in k, all_subdirectories))

    lrs, subdirectories = [], []
    for subdirectory in all_subdirectories:
        hit = re.search(r'lr_([^_|\/]*)', str(subdirectory))
        lr = float(hit.group(0).split('_')[1])
        # if lr <= 0.01:
        if lr <= 1000.:
            lrs.append(lr)
            subdirectories.append(subdirectory)
    lrs, subdirectories = list(zip(*sorted(zip(lrs, subdirectories), key=lambda x: x[0])))
    lrs = np.array(lrs)
    seed_ys = np.zeros((seeds.shape[0], lrs.shape[0])) - np.inf
    for i, subdirectory in enumerate(subdirectories):
        for seed in seeds:
            seed_path = Path(f"{subdirectory}/{seed}.json")
            key = str(seed_path).split('experiments/rl/')[1]
            data = cache.get(key)
            if data is None:
                with open(seed_path) as json_file:
                    data = json.load(json_file)
            if data.get('diverged', False):
                continue
            n_bins = 100
            bin_wid = data['n_samples'] // n_bins
            ts, rets, _, _, _, _ = bin_episodes(np.array(data['ts']), np.array(data['returns']), bin_wid)
            if not cache.has(key):
                data_copy = copy.deepcopy(data)
                data_copy.update({'ts': ts.tolist(), 'returns': rets.tolist()})
                cache.set(key, data_copy)
            seed_ys[seed, i] = np.array(rets).mean()

    # if filter_key == 'delta_0.001':
    #     print(seed_ys)
    return lrs, seed_ys

def plot_sensitivity(cache):
    optims = [
        # 'sgd', 'sgd_scaled', 'sgd_scaled_sqrt', 'adam', 'adam_scaled', 'adam_scaled_sqrt', 'adahessian', 'adahessian_scaled',
        # 'adahesscalegn', 'adahesscalegn_sqrt', 'adahesscalegn_adamstyle',
        # 'adahesscale', 'adahesscale_sqrt', 'adahesscale_adamstyle',
        # 'adahesscalegn_scaled', 'adahesscalegn_sqrt_scaled', 'adahesscalegn_adamstyle_scaled',
        # 'adahesscale_scaled', 'adahesscale_sqrt_scaled', 'adahesscale_adamstyle_scaled',

        # 'adahesscale_adamstyle_scaled', 'adahesscalegn_adamstyle_scaled',
        # 'adam'

        # 'adam', 'adam_scaled', 'adahesscale_adamstyle', 'adahesscale_adamstyle_scaled',

        'adam_hesscale', 'adam_scaled',
    ]

    # tasks = ['Ant', 'Walker2d', 'HalfCheetah', 'Hopper', 'InvertedDoublePendulum']
    tasks = ['InvertedPendulum', 'Swimmer', 'Reacher', 'Humanoid', 'HumanoidStandup']
    # tasks = ['Ant']
    # tasks = ['Hopper']
    # ylim = [-700., 4000.] if task != 'InvertedDoublePendulum' else [-700., 12000.]
    # ylim = [-700., 3000.]
    # ylim = [-700., 8000.]
    ylim = [-700., 2000.]

    # deltas = [.00001, .0001, .001]
    # deltas = [.000001, .00001, .0001]
    deltas = [.00000001, .00001]
    # deltas = [.00000001, .000001, .00001]

#     # A2C
#     exp_name = 'rl_a2c_3'
#     learner = 'a2c'
#     network = 'fcn_tanh_small'

    # PPO
    exp_name = 'rl_ppo_3'
    learner = 'ppo'
    network = 'var_net'

    seeds = np.arange(10)
    colors = [plt.get_cmap('tab10')(i) for i in np.arange(10)]
    for optim in optims:
        # if 'scaled' not in optim:
        #     continue

        print(optim)
        # n_cols, figsize = (3, (10., 2.4)) if 'scaled' in optim else (1, (7.767, 4.8))
        n_cols, figsize = (3, (10., 2.4)) if ('scaled' in optim and len(deltas) > 1) or optim == 'adam_hesscale' else (1, (7.767, 4.8))
        fig, axes = plt.subplots(1, n_cols)
        axes = np.array([axes]) if not isinstance(axes, np.ndarray) else axes.flatten()
        fig.set_size_inches(*figsize)
        for clr_id, task in enumerate(tasks):
            if 'ppo' in exp_name:
                task = task + '-v2'
            if 'scaled' in optim or optim == 'adam_hesscale':
                for ax, delta in zip(axes, deltas):
                    lrs, seed_ys = get_sensitivity(cache, exp_name, learner, network, optim, task, seeds, filter_key=f'delta_{delta}')
                    plot_line(ax, lrs, seed_ys, label=task, color=colors[clr_id])
                    ax.set_title(f'delta={delta}', fontdict={'fontsize': 10}, y=1.0)
            else:
                lrs, seed_ys = get_sensitivity(cache, exp_name, learner, network, optim, task, seeds)
                plot_line(axes[0], lrs, seed_ys, task, color=colors[clr_id])
        for ax in axes:
            ax.set_ylim(ylim)
            ax.set_xscale('log')
            ylabel = f'Return\naveraged over\nentire period\nand {seeds.shape[0]} runs'
            ax.set_ylabel(ylabel, labelpad=50, verticalalignment='center').set_rotation(0)
            ax.label_outer()
        leg_handles, leg_labels = axes[0].get_legend_handles_labels()
        fig.legend(leg_handles, leg_labels)#, loc=loc)
        fig.suptitle(f'{optim} - {learner}')
        fig.tight_layout()

        plt_pth = Path(f'plots/{exp_name}_sensit_largelr/{optim}_{tasks[0]}.pdf')
        plt_pth.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plt_pth, bbox_inches='tight')

def limits_union(l1, l2=None):
    if l2 is None:
        return l1
    else:
        return (np.minimum(l1[0], l2[0]), np.maximum(l1[1], l2[1]))

def plot_3_optim_sensitivity(cache, optim='a2c', legend=True, alltasks=False):
    optims = [
        'adam_hesscale', 'adahessian_hesscale_scaled', 'adahesscale_adamstyle_scaled',
    ]

    tasks = ['Ant', 'Walker2d', 'HalfCheetah', 'Hopper', 'Humanoid']
    if alltasks:
        tasks += ['InvertedPendulum', 'InvertedDoublePendulum', 'Swimmer', 'Reacher', 'HumanoidStandup']
    ylim = [-700., 2000.]

    delta = .00000001

    if optim == 'a2c':
        # A2C
        exp_name = 'rl_a2c_5'
        learner = 'a2c'
        network = 'fcn_tanh_small'
    else:
        # PPO
        exp_name = 'rl_ppo_3'
        learner = 'ppo'
        network = 'var_net'

    n_cols, figsize = (3, (10., 2.4))
    fig, axes = plt.subplots(1, n_cols)
    axes = np.array([axes]) if not isinstance(axes, np.ndarray) else axes.flatten()
    fig.set_size_inches(*figsize)

    seeds = np.arange(10)
    colors = [plt.get_cmap('tab10')(i) for i in np.arange(10)]
    for clr_id, task in enumerate(tasks):
        print(task)
        if 'ppo' in exp_name:
            task = task + '-v2'
        task_limits = None
        for ax, optim in zip(axes, optims):
            if optim == 'adahessian_hesscale_scaled': continue
            _, seed_ys = get_sensitivity(cache, exp_name, learner, network, optim, task, seeds, filter_key=f'delta_{delta}')
            ys = seed_ys.mean(axis=0)
            limits = (np.min(ys[np.logical_not(np.isinf(ys))]), np.max(ys[np.logical_not(np.isinf(ys))]))
            task_limits = limits_union(limits, task_limits)
        for ax, optim in zip(axes, optims):
            if optim == 'adahessian_hesscale_scaled': continue
            lrs, seed_ys = get_sensitivity(cache, exp_name, learner, network, optim, task, seeds, filter_key=f'delta_{delta}')
            seed_ys = (seed_ys - task_limits[0]) / (task_limits[1] - task_limits[0])
            plot_line(ax, lrs, seed_ys, label=task, color=colors[clr_id])
            ax.set_title(f'{optim}', fontdict={'fontsize': 10}, y=1.05)

    for ax in axes:
        ax.set_ylim((-.1, 1.2))
        ax.set_xscale('log')
        # ylabel = f'Return\naveraged over\nentire period\nand {seeds.shape[0]} runs'
        # ax.set_ylabel(ylabel, labelpad=50, verticalalignment='center').set_rotation(0)
        ax.set_ylabel('Average Return')
        ax.set_xlabel('Learning Rate')
        ax.label_outer()
        ax.tick_params(labelsize=6)
        ax.xaxis.get_offset_text().set_fontsize(6)
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.yaxis.set_offset_position('left')

    if legend:
        leg_handles, leg_labels = axes[0].get_legend_handles_labels()
        fig.legend(leg_handles, leg_labels)#, loc=loc)
    # fig.suptitle(f'{optim} - {learner}')
    fig.tight_layout()

    algo = 'a2c' if 'a2c' in exp_name else 'ppo'
    plt_pth = Path(f'plots/{exp_name}_sensit_3_optims/{algo}_legend_{legend}_alltasks_{alltasks}.pdf')
    plt_pth.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plt_pth, bbox_inches='tight')

def get_ylim(task):
    task_id = task.split('-v2')[0]
    if task_id in ['Ant', 'Walker2d', 'HalfCheetah', 'Hopper', 'Humanoid']:
        return [-700, 4000]
    else:
        return {
            'InvertedPendulum': [-50, 1100],
            'InvertedDoublePendulum': [-700, 12000],
            'Swimmer': [0, 130],
            'Reacher': [-70, 10],
            'HumanoidStandup': [0, 200000],
        }[task_id]

def get_returns(cache, subdir):
    # seeds = os.listdir(f'{subdir}')
    # ts_list, rets_list = [], []
    n_bins = 50
    xs, seed_ys = None, np.zeros((10, n_bins))
    # for seed in seeds:
    for seed in np.arange(10):
        key = f'{subdir}/{seed}.json'.split('experiments/rl/')[1]
        data = cache.get(key)
        if data is None:
            with open(f"{subdir}/{seed}") as json_file:
                data = json.load(json_file)
        if data.get('diverged', False):
            return None
        bin_wid = data['n_samples'] // n_bins
        ts, rets, _, _, _, _ = bin_episodes(np.array(data['ts']), np.array(data['returns']), bin_wid)
        if not cache.has(key):
            data_copy = copy.deepcopy(data)
            data_copy.update({'ts': ts.tolist(), 'returns': rets.tolist()})
            cache.set(key, data_copy)

        if xs is None:
            xs = ts[:n_bins]
        seed_ys[seed] = rets[:n_bins]
        # ts_list.append(ts[:n_bins])
        # rets_list.append(rets[:n_bins])
    return xs, seed_ys

def get_optim_color(optim):
    noscale_opt = optim.split('_scaled')[0]
    return {
        'adam': 'tab:pink',
        'adahessian': 'tab:brown',
        'adahesscale_adamstyle': 'tab:green',
        'adahesscalegn_adamstyle': 'tab:orange',
        'adam_hesscale': 'tab:blue',
    }[noscale_opt]

def plot_4_optims(cache, optim='a2c', scaled=False, gn=True, legend=True):
    if gn:
        optims_list = [
            # ['sgd', 'adam', 'adahessian', 'adahesscale_adamstyle',],
            # ['sgd_scaled', 'adam_scaled', 'adahessian_scaled', 'adahesscale_adamstyle_scaled',],

            ['adam', 'adahessian', 'adahesscale_adamstyle', 'adahesscalegn_adamstyle'],
            ['adam_scaled', 'adam_hesscale', 'adahessian_scaled', 'adahesscale_adamstyle_scaled', 'adahesscalegn_adamstyle_scaled'],
        ]
    else:
        optims_list = [
            ['adam', 'adahessian', 'adahesscale_adamstyle'],
            ['adam_scaled', 'adam_hesscale', 'adahessian_scaled', 'adahesscale_adamstyle_scaled'],
        ]

    # tasks = ['Ant', 'Walker2d', 'HalfCheetah', 'Hopper', 'InvertedDoublePendulum']
    tasks = ['Ant', 'Walker2d', 'HalfCheetah', 'Hopper', 'InvertedDoublePendulum', 'InvertedPendulum', 'Swimmer', 'Reacher', 'Humanoid', 'HumanoidStandup']

    delta = .00000001

    if optim == 'a2c':
        # A2C
        exp_name = 'rl_a2c_5'
        learner = 'a2c'
        network = 'fcn_tanh_small'
    else:
        # PPO
        exp_name = 'rl_ppo_3'
        learner = 'ppo'
        network = 'var_net'

    filter = False
    optims = optims_list[int(scaled)]

    seeds = np.arange(10)
    colors = [plt.get_cmap('tab10')(i) for i in np.arange(10)]
    # n_cols, figsize = (3, (10., 2.4)) if 'scaled' in optim else (1, (7.767, 4.8))
    figsize = (10., 4.8)
    fig, axes = plt.subplots(2, 5)
    axes = np.array([axes]) if not isinstance(axes, np.ndarray) else axes.flatten()
    # fig.delaxes(axes[-1])
    fig.set_size_inches(*figsize)
    for ax, task in zip(axes, tasks):
        print(task)
        if 'ppo' in exp_name:
            task = task + '-v2'
        for clr_id, optim in enumerate(optims):
            # lrs, seed_ys = get_sensitivity(cache, exp_name, learner, network, optim, task, seeds, filter_key=f'delta_{delta}')
            filter_key = None
            if filter and 'adahesscale' in optim:
                filter_key = 'delta_1e-08_eps_1e-05_lr_0.0003' if 'scaled' in optim else 'eps_1e-05_lr_0.0003'
            best_runs = core.best_config.BestConfig(exp_name, task, network, [learner], [optim]).get_best_run(cache, measure='returns', filter_key=filter_key)
            print(best_runs)
            ts, seed_ys = get_returns(cache, best_runs[0])
            color = get_optim_color(optim)
            plot_line(ax, ts, seed_ys, label=optim, color=color)

        ax.set_title(f'{task}', fontdict={'fontsize': 10}, y=1.05)
        # ylim = [-700., 4000.] if 'InvertedDoublePendulum' not in task else [-700., 12000.]
        ylim = get_ylim(task)
        ax.set_ylim(ylim)
        # ylabel = f'Return\naveraged over\nentire period\nand {seeds.shape[0]} runs'
        # ax.set_ylabel(ylabel, labelpad=50, verticalalignment='center').set_rotation(0)
        if ax in axes[[0, 5]]:
            ax.set_ylabel(f'Average Return')
        if ax in axes[5:]:
            ax.set_xlabel('Time Step')
        # ax.label_outer()
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        ax.tick_params(labelsize=6)
        ax.xaxis.get_offset_text().set_fontsize(6)
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.yaxis.set_offset_position('left')

    if legend:
        leg_handles, leg_labels = axes[0].get_legend_handles_labels()
        fig.legend(leg_handles, leg_labels, loc='lower left')
    # fig.suptitle(f'{learner} - {"scaled" if scaled else "nonscaled"}')
    fig.tight_layout()

    algo = 'a2c' if 'a2c' in exp_name else 'ppo'
    plt_pth = Path(f'plots/{exp_name}_four_optims/{algo}_{"scaled" if scaled else "nonscaled"}{"_filter" if filter else ""}_{len(tasks)}_gn_{gn}_legend_{legend}.pdf')
    plt_pth.parent.mkdir(parents=True, exist_ok=True)
    # fig.savefig(plt_pth, bbox_inches='tight')


def main():
    cache = Cache(Path('logs.json'))
    # plot_learning_curves(cache)
    # plot_sensitivity(cache)

    # plot_4_optims(cache, optim='a2c', scaled=False)
    plot_4_optims(cache, optim='a2c', scaled=True)
    # plot_4_optims(cache, optim='ppo', scaled=False)
    plot_4_optims(cache, optim='ppo', scaled=True)

    # for legend in [True, False]:
    #     plot_4_optims(cache, optim='a2c', scaled=False, gn=True, legend=legend)
    #     plot_4_optims(cache, optim='ppo', scaled=False, gn=True, legend=legend)
    #     plot_4_optims(cache, optim='a2c', scaled=False, gn=False, legend=legend)
    #     plot_4_optims(cache, optim='ppo', scaled=False, gn=False, legend=legend)

    # for legend in [True, False]:
    #     plot_3_optim_sensitivity(cache, optim='a2c', legend=legend, alltasks=False)
    #     plot_3_optim_sensitivity(cache, optim='a2c', legend=legend, alltasks=True)
    #     plot_3_optim_sensitivity(cache, optim='ppo', legend=legend, alltasks=False)
    #     plot_3_optim_sensitivity(cache, optim='ppo', legend=legend, alltasks=True)

if __name__ == "__main__":
    main()
