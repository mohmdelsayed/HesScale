class ParamsSensitivity:
    def __init__(self, task_name, metric, network_name, learners, sensitivity_param):
        self.task_name = task_name
        self.metric = metric
        self.network_name = network_name
        self.learners = learners
        self.sensitivity_param = sensitivity_param
    
    def get_sensitivity(self, measure="accuracies"):
        raise NotImplementedError

if __name__ == "__main__":    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--network_name", type=str, required=True)
    parser.add_argument('--metric', nargs='+', default=[])
    parser.add_argument('--learners', nargs='+', default=[])
    parser.add_argument("--sensitivity_param", type=str, required=True)
    args = parser.parse_args()
    param_sensitivity = ParamsSensitivity(args.task_name, args.metric, args.network_name, args.learners, args.sensitivity_param)
    param_sensitivity.get_sensitivity(measure="accuracies")