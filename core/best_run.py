import os
import json
import numpy as np
class BestRun:
    def __init__(self, task_name, metric, network_name, learners):
        self.task_name = task_name
        self.metric = metric
        self.network_name = network_name
        self.learners = learners
    
    def get_best_run(self, measure="losses", filter_key=None):
        best_configs = []
        for learner in self.learners:
            path = f"logs/{self.task_name}/{learner}/{self.network_name}/"
            subdirectories = [f.path for f in os.scandir(path) if f.is_dir()]

            configs = {}
            # loop over all hyperparameters configurations
            if filter_key is not None:
                subdirectories = filter(lambda k: filter_key in k, subdirectories)
            for subdirectory in subdirectories:
                configs[subdirectory] = {}
                seeds = os.listdir(f'{subdirectory}')
                configuration_list = []
                for seed in seeds:
                    with open(f"{subdirectory}/{seed}") as json_file:
                        data = json.load(json_file)
                        configuration_list.append(data[measure])

                mean_list = np.nan_to_num(np.array(configuration_list), nan=np.iinfo(np.int32).max).mean(axis=-1)
                configs[subdirectory]["means"] = mean_list
            if measure == "losses":
                best_configs.append(min(configs.keys(), key=(lambda k: sum(configs[k]["means"]))))
            elif measure == "accuracies":
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
    best_configs = BestRun(args.task_name, args.metric, args.network_name, args.learners).get_best_run(measure="losses")
    print(best_configs)
