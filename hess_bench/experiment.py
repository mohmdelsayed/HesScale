import os
import pickle
import time

import numpy as np
import torch

from hess_bench.data_generator import TargetGenerator
from hess_bench.hess_comparision import HessComp


class HessExperimentComparision:
    def __init__(
        self,
        configs,
        seed,
    ):
        super(HessExperimentComparision, self).__init__()

        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.data_generator = TargetGenerator(**configs["data_generator_params"])
        configs["predictor_params"]["n_classes"] = configs["data_generator_params"][
            "out_size"
        ]
        configs["predictor_params"]["n_obs"] = configs["data_generator_params"][
            "in_size"
        ]
        configs["predictor_params"]["lr"] = configs["lr"]
        if configs["data_generator_params"]["task"] == "regression":
            self.predictor = HessComp(**configs["predictor_params"])
        else:
            self.predictor = HessComp(**configs["predictor_params"])

        self.dataset_size = configs["data_generator_params"]["dataset_size"]

    def train(self, lamda=1.0):
        # inputs, labels = self.data_generator.get_dataset(dataset_size=self.dataset_size)

        # for (inp, label) in iterate_minibatches(inputs, labels, batchsize=50):
        #     self.predictor.learn(inp, label)

        inputs, labels = self.data_generator.get_dataset(dataset_size=self.dataset_size)

        avg_exact_hessian, avg_exact_grad = self.predictor.avg_exact_hess(
            inputs, labels
        )
        avgs_lists = {}

        for (inp, label) in iterate_minibatches(inputs, labels, batchsize=50):

            sample_error_methods = self.predictor.compare_hess_methods(
                inp, label, avg_exact_hessian, avg_exact_grad, lamda=lamda
            )

            for method in sample_error_methods:
                if not method in avgs_lists:
                    avgs_lists[method] = {}
                for name in sample_error_methods[method]:
                    if not name in avgs_lists[method]:
                        sums = 0
                        avgs_lists[method][name] = []

                    avgs_lists[method][name].append(sample_error_methods[method][name])

        # print(f"Finished run with seed: {self.seed}")

        return avgs_lists


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0], batchsize):
        end_idx = min(start_idx + batchsize, inputs.shape[0])
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield inputs[excerpt], targets[excerpt]
