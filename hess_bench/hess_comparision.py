import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from backpack import backpack, extend
from backpack.extensions import DiagGGNExact, DiagGGNMC, DiagHessian
from hesscale import HesScale
from torch.optim import SGD


class HessComp(nn.Module):
    def __init__(
        self,
        n_classes,
        n_obs,
        hidden_units,
        lr,
    ):

        super(HessComp, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=2),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2),
            
            torch.nn.Conv2d(1, 1, kernel_size=2),
            torch.nn.Tanh(),
            
            torch.nn.Conv2d(1, 1, kernel_size=2),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.MaxPool2d(kernel_size=2),
            
            torch.nn.Flatten(),
            torch.nn.Linear(1, 10),
        )
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(n_obs, hidden_units),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(hidden_units, hidden_units),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(hidden_units, hidden_units),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(hidden_units, hidden_units),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(hidden_units, hidden_units),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(hidden_units, hidden_units),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(hidden_units, hidden_units),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(hidden_units, hidden_units),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(hidden_units, hidden_units),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(hidden_units, n_classes),
        # )

        self.model_softmax = nn.Sequential(
            copy.deepcopy(self.model), torch.nn.LogSoftmax(dim=1)
        )
        # self.model_softmax = nn.Sequential(
        #     copy.deepcopy(self.model)
        # )
        extend(self.model)
        extend(self.model_softmax)
        self.loss_func = nn.CrossEntropyLoss(reduction="mean")

        self.loss_func_log = nn.NLLLoss(reduction="mean")
        # self.loss_func_log = nn.CrossEntropyLoss(reduction="mean")

        extend(self.loss_func)
        extend(self.loss_func_log)

        self.optimizer = SGD(
            list(self.model.parameters()),
            lr=1e-3,
        )

        self.optimizer_softmax = SGD(
            list(self.model_softmax.parameters()),
            lr=1e-3,
        )

    def compare_hess_methods(
        self, state, label, avg_exact_hess, avg_exact_grad, lamda=1.0
    ):

        state = torch.from_numpy(state).float()
        label = torch.tensor(label).long()

        loss = self.loss_func(self.model(state), label)
        loss_softmax = self.loss_func_log(self.model_softmax(state), label)

        self.optimizer.zero_grad()
        with backpack(
            HesScale(), DiagGGNMC(mc_samples=1), DiagGGNExact(), DiagHessian()
        ):
            loss.backward()

        self.optimizer_softmax.zero_grad()
        with backpack(HesScale()):
            loss_softmax.backward()

        summed_errors = {}

        summed_errors["HesScale"] = {}
        summed_errors["GGN"] = {}
        summed_errors["GGN_MC"] = {}

        summed_errors["OBD"] = {}
        summed_errors["|H|"] = {}

        # summed_errors["SGD_H"] = 0.0
        summed_errors["H"] = {}

        for (name, param), (name_soft, param_soft) in zip(
            self.model.named_parameters(), self.model_softmax.named_parameters()
        ):
            x = param.diag_h.data.clone()  # avg_exact_hess[name]#
            summed_errors["HesScale"][name] = (
                torch.abs(x - lamda * param.hesscale.clone()).sum().item()
            )
            summed_errors["GGN_MC"][name] = (
                torch.abs(x - lamda * param.diag_ggn_mc.clone()).sum().item()
            )
            summed_errors["GGN"][name] = (
                torch.abs(x - lamda * param.diag_ggn_exact.clone()).sum().item()
            )

            summed_errors["H"][name] = (
                torch.abs(x - lamda * param.diag_h.data.clone()).sum().item()
            )
            summed_errors["OBD"][name] = (
                torch.abs(x - lamda * param_soft.hesscale.clone()).sum().item()
            )
            summed_errors["|H|"][name] = torch.abs(x - 0.0).sum().item()

            # summed_errors["SGD_H"] += torch.abs(avg_exact_hess[name] - lamda * 1.0).sum().item()

        return summed_errors

    def avg_exact_hess(self, states, labels):
        exact_hessian, exact_grad = {}, {}
        states = torch.from_numpy(states).float()
        labels = torch.tensor(labels).long()

        self.optimizer.zero_grad()
        loss = self.loss_func(self.model(states), labels)
        with backpack(DiagHessian()):
            loss.backward()

        for name, param in self.model.named_parameters():
            exact_hessian[name] = param.diag_h.data.clone()
            exact_grad[name] = param.grad.data.clone()

        return exact_hessian, exact_grad

    def learn(self, state, label):
        label = torch.tensor(label).long()
        state = torch.tensor(state).float()
        self.optimizer.zero_grad()
        loss = self.loss_func(self.model(state), label)

        # with backpack():
        loss.backward()

        self.optimizer.step()
        return loss.item()
