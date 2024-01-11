import json
from pathlib import Path


class Logger:
    """
    This class is responsible for logging the training process.
    Log file will be saved in the log directory with JSON format.
    """

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir

    def get_log_path(self, **kwargs):
        opt_hps = kwargs['optimizer_hps']
        file_name = '_'.join([f'{key}_{opt_hps[key]}' for key in sorted(opt_hps.keys())])
        
        path = Path(f"{self.log_dir}/{kwargs['exp_name']}/{kwargs['task']}/{kwargs['learner']}/{kwargs['optimizer']}/{kwargs['network']}/{file_name}/{kwargs['seed']}.json")
        return path

    def log(self, **kwargs):
        json_object = json.dumps(kwargs, indent=4)
        path = self.get_log_path(**kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as outfile:
            outfile.write(json_object)

