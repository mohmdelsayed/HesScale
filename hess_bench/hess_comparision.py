import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from backpack import backpack, extend
from backpack.extensions import DiagGGNExact, DiagGGNMC, DiagHessian, KFAC
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

        # self.model = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, 1, kernel_size=2),
        #     torch.nn.Tanh(),
        #     torch.nn.MaxPool2d(kernel_size=2),
            
        #     torch.nn.Conv2d(1, 1, kernel_size=2),
        #     torch.nn.Tanh(),
            
        #     torch.nn.Conv2d(1, 1, kernel_size=2),
        #     torch.nn.Tanh(),
        #     torch.nn.MaxPool2d(kernel_size=2),
        #     torch.nn.MaxPool2d(kernel_size=2),
            
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(1, 10),
        # )
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_obs, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, n_classes),
        )

        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(n_obs, hidden_units),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(hidden_units, hidden_units),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_units, hidden_units),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(hidden_units, hidden_units),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_units, hidden_units),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(hidden_units, hidden_units),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_units, hidden_units),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(hidden_units, hidden_units),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_units, hidden_units),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(hidden_units, n_classes),
        # )
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
            lr=1e-3,
        )

        self.optimizer_softmax = SGD(
            list(self.model_softmax.parameters()),
            lr=1e-3,
        )

    def compare_hess_methods(
        self, state, label, lamda=1.0
    ):

        state = torch.from_numpy(state).float()
        label = torch.tensor(label).long()

        loss = self.loss_func(self.model(state), label)
        loss_softmax = self.loss_func_log(self.model_softmax(state), label)

        self.optimizer.zero_grad()
        with backpack(
            HesScale(), DiagGGNMC(mc_samples=1), DiagGGNExact(), DiagHessian(), KFAC(mc_samples=1)
        ):
            loss.backward(create_graph=True)

        params = []
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)
        hut_traces = self.get_adahess_estimate(params, grads, mc_samples=50)

        self.optimizer_softmax.zero_grad()
        with backpack(HesScale()):
            loss_softmax.backward()

        summed_errors = {}

        summed_errors["HesScale"] = {}
        summed_errors["GGN"] = {}
        summed_errors["GGN_MC"] = {}

        summed_errors["OBD"] = {}
        summed_errors["|H|"] = {}

        summed_errors["AdaHess"] = {}
        summed_errors["KFAC"] = {}
        summed_errors["H"] = {}

        for (name, param), (name_soft, param_soft), adahess_diag in zip(
            self.model.named_parameters(), self.model_softmax.named_parameters(), hut_traces
        ):
            x = param.diag_h.data.clone()
            kfac_estimate = self.get_kfac_estimate(param.kfac, param)
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

            summed_errors["AdaHess"][name] = (
                torch.abs(x - lamda * adahess_diag.clone()).sum().item()
            )
            summed_errors["KFAC"][name] = (
                torch.abs(x - lamda * kfac_estimate).sum().item()
            )
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

    def get_adahess_estimate(self, params, grads, mc_samples=1):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """
        self.single_gpu = True

        # Check backward was called with create_graph set to True
        for i, grad in enumerate(grads):
            if grad.grad_fn is None:
                raise RuntimeError('Gradient tensor {:} does not have grad_fn. When calling\n'.format(i) +
                           '\t\t\t  loss.backward(), make sure the option create_graph is\n' +
                           '\t\t\t  set to True.')

        hvs_sum = None
        for _ in range(mc_samples):
            
            v = [2 * torch.randint_like(p, high=2) - 1 for p in params]

            # this is for distributed setting with single node and multi-gpus, 
            # for multi nodes setting, we have not support it yet.
            if not self.single_gpu:
                for v1 in v:
                    dist.all_reduce(v1)
            if not self.single_gpu:
                for v_i in v:
                    v_i[v_i < 0.] = -1.
                    v_i[v_i >= 0.] = 1.

            if hvs_sum is None:
                hvs_sum = [x * (1/mc_samples) for x in torch.autograd.grad(
                grads,
                params,
                grad_outputs=v,
                only_inputs=True,
                retain_graph=True)] 
            else:
                hvs_sum += [x * (1/mc_samples) for x in torch.autograd.grad(
                grads,
                params,
                grad_outputs=v,
                only_inputs=True,
                retain_graph=True)] 
                
                
        hutchinson_trace = []
        for hv in hvs_sum:
            param_size = hv.size()
            if len(param_size) <= 2:  # for 0/1/2D tensor
                # Hessian diagonal block size is 1 here.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = hv.abs()

            elif len(param_size) == 4:  # Conv kernel
                # Hessian diagonal block size is 9 here: torch.sum() reduces the dim 2/3.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = torch.mean(hv.abs(), dim=[2, 3], keepdim=True)
            hutchinson_trace.append(tmp_output)

        # this is for distributed setting with single node and multi-gpus, 
        # for multi nodes setting, we have not support it yet.
        if not self.single_gpu:
            for output1 in hutchinson_trace:
                dist.all_reduce(output1 / torch.cuda.device_count())
        
        return hutchinson_trace
    
    def get_kfac_estimate(self, curv_p, param):
        if len(curv_p) == 1:
            kfac = curv_p[0]
            return torch.diagonal(kfac, 0)
        else:
            kfac1, kfac2 = curv_p
            mtx = torch.kron(kfac1, kfac2)
            return torch.diagonal(mtx, 0).view_as(param)