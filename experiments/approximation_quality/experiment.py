import copy
import numpy as np
import torch
import torch.nn as nn
from backpack import backpack, extend
from backpack.extensions import DiagGGNExact, DiagGGNMC, DiagHessian, KFAC
from hesscale import HesScale, HesScaleGN
from torch.optim import SGD
import torchvision
from utils import get_kfac_estimate, get_adahess_estimate

activation_func = {
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "sigmoid": torch.nn.Sigmoid,
}

DiagGGNMC_1 = DiagGGNMC(mc_samples=1)
DiagGGNMC_1.savefield = "diag_ggn_mc_1"
DiagGGNMC_50 = DiagGGNMC(mc_samples=50)
DiagGGNMC_50.savefield = "diag_ggn_mc_50"
KFAC_1 = KFAC(mc_samples=1)
KFAC_1.savefield = "kfac_1"
KFAC_50 = KFAC(mc_samples=50)
KFAC_50.savefield = "kfac_50"
methods = {
    "HS": "hesscale",
    "BL89": None,
    "AdaHess_1": None,
    # "AdaHess_50": None,

    # "GGN": "diag_ggn_exact",
    "HSGGN": "hesscale_gn",
    "GGNMC_1": "diag_ggn_mc_1",
    # "GGNMC_50": "diag_ggn_mc_50",
    "KFAC_1": None,
    # "KFAC_50": None,
    "g2": None,
    "H": "diag_h",
    "|H|": None,
}


class HessQualityExperiment:
    def __init__(
        self,
        configs,
        seed,
    ):
        super(HessQualityExperiment, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.dataset_size = configs["data_generator_params"]["dataset_size"]
        self.batch_size = configs["data_generator_params"]["batch_size"]
        self.approximator = HessApprox(
            n_classes=configs["data_generator_params"]["out_size"],
            network_shape=configs["predictor_params"]["network_hidden_units"],
            act=configs["predictor_params"]["activation_func"],
            n_obs=configs["data_generator_params"]["in_size"],
            lr=configs["lr"],
        )

    def train(self):

        avgs_lists = {}

        trainset = torchvision.datasets.MNIST('dataset/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ]))
                                    
        indx = list(range(self.dataset_size))
        trainset_sub = torch.utils.data.Subset(trainset, indx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        train_loader_ = torch.utils.data.DataLoader(trainset_sub, batch_size=self.batch_size, shuffle=True)

        # train on full dataset twice
        for i, (inp, label) in enumerate(train_loader):
            loss = self.approximator.learn(inp, label)
        for i, (inp, label) in enumerate(train_loader):
            loss = self.approximator.learn(inp, label)

        for i, (inp, label) in enumerate(train_loader_):

            sample_error_per_method = self.approximator.compare_hess_methods(
                inp, label
            )
            print("Element", i)
    
            # self.approximator.learn(inp, label)

            if not avgs_lists:
                avgs_lists = sample_error_per_method.copy()
            else:
                for method in sample_error_per_method:
                    for name in sample_error_per_method[method]:
                        if type(avgs_lists[method][name]) is not list:
                            avgs_lists[method][name] = [
                                avgs_lists[method][name],
                                sample_error_per_method[method][name],
                            ]
                        else:
                            avgs_lists[method][name].append(
                                sample_error_per_method[method][name]
                            )

        return avgs_lists

class HessApprox(nn.Module):
    def __init__(
        self,
        n_classes,
        n_obs,
        lr=1e-4,
        network_shape=[128, 128, 128],
        act="tanh",
    ):

        super(HessApprox, self).__init__()

        if act not in activation_func:
            raise "Not available activation function"

        self.model = self.create_network(n_obs, network_shape, n_classes, act)

        self.model_softmax = nn.Sequential(
            copy.deepcopy(self.model), torch.nn.LogSoftmax(dim=1)
        )
        extend(self.model)
        extend(self.model_softmax)
        self.loss_func = nn.CrossEntropyLoss(reduction="mean")

        self.loss_func_log = nn.NLLLoss(reduction="mean")

        extend(self.loss_func)
        extend(self.loss_func_log)

        self.optimizer = SGD(
            list(self.model.parameters()),
            lr=lr,
        )

        self.optimizer_softmax = SGD(
            list(self.model_softmax.parameters()),
            lr=lr,
        )

    def compare_hess_methods(self, state, label):

        label = torch.tensor(label).long()
        state = torch.flatten(state, start_dim=1)

        loss = self.loss_func(self.model(state), label)
        loss_softmax = self.loss_func_log(self.model_softmax(state), label)

        self.optimizer.zero_grad()
        with backpack(
            DiagGGNMC_1,
            DiagGGNMC_50,
            KFAC_1,
            KFAC_50,
            HesScale(),
            HesScaleGN(),
            DiagGGNExact(),
            DiagHessian(),
        ):
            loss.backward(create_graph=True)

        adahess_diags_1 = get_adahess_estimate(
            self.model.parameters(), mc_samples=1
        )
        adahess_diags_50 = get_adahess_estimate(
            self.model.parameters(), mc_samples=50
        )

        self.optimizer_softmax.zero_grad()
        with backpack(HesScale()):
            loss_softmax.backward()

        summed_errors = {}

        for method in methods:
            summed_errors[method] = {}

        for (name, param), (name_soft, param_soft), adahess_diag_1, adahess_diag_50 in zip(
            self.model.named_parameters(),
            self.model_softmax.named_parameters(),
            adahess_diags_1,
            adahess_diags_50
        ):
            exact_h_diagonals = param.diag_h.data.data
            kfac_estimate_1 = get_kfac_estimate(param.kfac_1, param)
            kfac_estimate_50 = get_kfac_estimate(param.kfac_50, param)

            for method in methods:
                if method == "|H|":
                    summed_errors[method][name] = (
                        torch.abs(exact_h_diagonals - 0.0).sum().item()
                    )
                elif method == "g2":
                    summed_errors[method][name] = (
                        torch.abs(exact_h_diagonals - param.grad.data ** 2)
                        .sum()
                        .item()
                    )
                elif method == "KFAC_1":
                    summed_errors[method][name] = (
                        torch.abs(exact_h_diagonals - kfac_estimate_1.data)
                        .sum()
                        .item()
                    )
                elif method == "KFAC_50":
                    summed_errors[method][name] = (
                        torch.abs(exact_h_diagonals - kfac_estimate_50.data)
                        .sum()
                        .item()
                    )
                elif method == "AdaHess_50":
                    summed_errors[method][name] = (
                        torch.abs(exact_h_diagonals - adahess_diag_50.data)
                        .sum()
                        .item()
                    )
                elif method == "AdaHess_1":
                    summed_errors[method][name] = (
                        torch.abs(exact_h_diagonals - adahess_diag_1.data)
                        .sum()
                        .item()
                    )
                elif method == "BL89":
                    summed_errors[method][name] = (
                        torch.abs(exact_h_diagonals - param_soft.hesscale.data)
                        .sum()
                        .item()
                    )
                else:
                    summed_errors[method][name] = (
                        torch.abs(exact_h_diagonals - getattr(param, methods[method]).data)
                        .sum()
                        .item()
                    )

        return summed_errors

    def learn(self, state, label):
        label = torch.tensor(label).long()
        state = torch.tensor(state).float()
        state = torch.flatten(state, start_dim=1)

        self.optimizer.zero_grad()
        self.optimizer_softmax.zero_grad()

        loss = self.loss_func(self.model(state), label)
        loss_softmax = self.loss_func_log(self.model(state), label)

        loss.backward()
        loss_softmax.backward()

        self.optimizer.step()
        self.optimizer_softmax.step()
        return loss

    def create_network(self, n_obs, network_shape, n_classes, act):
        network = []
        index = 0
        network_shape = [n_obs, *network_shape, n_classes]
        while index < len(network_shape) - 1:
            network.append(
                torch.nn.Linear(
                    network_shape[index], network_shape[index + 1], bias=True
                )
            )
            network.append(activation_func[act]())
            index += 1
        network.pop(-1)
        return torch.nn.Sequential(*network)
