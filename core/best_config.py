import os
import json
import numpy as np
from torch import sub

import core.plot.rl_plotter

class BestConfig:
    def __init__(self, exp_name, task_name, network_name, learners, optims):
        self.exp_name = exp_name
        self.task_name = task_name
        self.network_name = network_name
        self.learners = learners
        self.optims = optims
    
    def get_best_run(self, measure="losses", filter_key=None):
        best_configs = []
        for learner, optim in zip(self.learners, self.optims):
            path = f"logs/{self.exp_name}/{self.task_name}/{learner}/{optim}/{self.network_name}/"
            subdirectories = [f.path for f in os.scandir(path) if f.is_dir()]

            configs = {}
            # loop over all hyperparameters configurations
            if filter_key is not None:
                subdirectories = filter(lambda k: filter_key in k, subdirectories)
            for subdirectory in subdirectories:
                seeds = os.listdir(f'{subdirectory}')
                configuration_list = []
                ts_list = []
                diverged = False
                for seed in seeds:
                    with open(f"{subdirectory}/{seed}") as json_file:
                        data = json.load(json_file)
                        diverged = data.get('diverged', False)
                        if diverged: break
                        if measure == 'returns':
                            n_bins = 100
                            bin_wid = data['n_samples'] // n_bins
                            ts, rets, _, _, _, _ = core.plot.rl_plotter.bin_episodes(np.array(data['ts']), np.array(data[measure]), bin_wid)
                            ts_list.append(ts[:n_bins])
                            configuration_list.append(rets[:n_bins])
                        else:
                            configuration_list.append(data[measure])

                if not diverged:
                    mean_list = np.nan_to_num(np.array(configuration_list), nan=np.iinfo(np.int32).max).mean(axis=-1)
                    configs[subdirectory] = {'ts': ts_list[0], 'means': mean_list}
            if measure == "losses":
                best_configs.append(min(configs.keys(), key=(lambda k: sum(configs[k]["means"]))))
            elif measure in ["accuracies", "returns"]:
                best_configs.append(max(configs.keys(), key=(lambda k: sum(configs[k]["means"]))))
            else:
                raise Exception("measure must be loss or accuracy")
        return best_configs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--network_name", type=str, required=True)
    parser.add_argument('--metric', nargs='+', default=[])
    parser.add_argument('--learners', nargs='+', default=[])
    args = parser.parse_args()
    best_configs = BestConfig(args.task_name, args.metric, args.network_name, args.learners).get_best_run(measure="losses")
    print(best_configs)
