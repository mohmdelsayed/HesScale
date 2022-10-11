import copy
import numpy as np
import torch
import torch.nn as nn
from backpack import backpack, extend
from backpack.extensions import DiagGGNExact, DiagGGNMC, DiagHessian, KFAC
from experiments.approximation_quality.data_generator import TargetGenerator
from experiments.approximation_quality.act_func import activation_func
from hesscale import HesScale, HesScaleGGN
from torch.optim import SGD

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
    "AdaHess_50": None,

    "GGN": "diag_ggn_exact",
    "HSGGN": "hesscale_ggn",
    "GGNMC_1": "diag_ggn_mc_1",
    "GGNMC_50": "diag_ggn_mc_50",
    "KFAC_1": None,
    "KFAC_50": None,
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
        self.data_generator = TargetGenerator(**configs["data_generator_params"])
        self.approximator = HessApprox(
            n_classes=configs["data_generator_params"]["out_size"],
            network_shape=configs["predictor_params"]["network_hidden_units"],
            act=configs["predictor_params"]["activation_func"],
            n_obs=configs["data_generator_params"]["in_size"],
            lr=configs["lr"],
        )

    def train(self, lamda=1.0):

        avgs_lists = {}

        inputs, labels = self.data_generator.get_dataset(dataset_size=self.dataset_size)

        for (inp, label) in self.iterate_minibatches(
            inputs, labels, batchsize=self.batch_size
        ):
            sample_error_per_method = self.approximator.compare_hess_methods(
                inp, label, lamda=lamda
            )

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

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
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

    def compare_hess_methods(self, state, label, lamda=1.0):

        state = torch.from_numpy(state).float()
        label = torch.tensor(label).long()

        loss = self.loss_func(self.model(state), label)
        loss_softmax = self.loss_func_log(self.model_softmax(state), label)

        self.optimizer.zero_grad()
        with backpack(
            DiagGGNMC_1,
            DiagGGNMC_50,
            KFAC_1,
            KFAC_50,
            HesScale(),
            HesScaleGGN(),
            DiagGGNExact(),
            DiagHessian(),
        ):
            loss.backward(create_graph=True)

        adahess_diags_1 = self.get_adahess_estimate(
            self.model.parameters(), mc_samples=1
        )
        adahess_diags_50 = self.get_adahess_estimate(
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
            kfac_estimate_1 = self.get_kfac_estimate(param.kfac_1, param)
            kfac_estimate_50 = self.get_kfac_estimate(param.kfac_50, param)

            for method in methods:
                if method == "|H|":
                    summed_errors[method][name] = (
                        torch.abs(exact_h_diagonals - 0.0).sum().item()
                    )
                elif method == "g2":
                    summed_errors[method][name] = (
                        torch.abs(exact_h_diagonals - lamda * param.grad.data ** 2)
                        .sum()
                        .item()
                    )
                elif method == "KFAC_1":
                    summed_errors[method][name] = (
                        torch.abs(exact_h_diagonals - lamda * kfac_estimate_1.data)
                        .sum()
                        .item()
                    )
                elif method == "KFAC_50":
                    summed_errors[method][name] = (
                        torch.abs(exact_h_diagonals - lamda * kfac_estimate_50.data)
                        .sum()
                        .item()
                    )
                elif method == "AdaHess_50":
                    summed_errors[method][name] = (
                        torch.abs(exact_h_diagonals - lamda * adahess_diag_50.data)
                        .sum()
                        .item()
                    )
                elif method == "AdaHess_1":
                    summed_errors[method][name] = (
                        torch.abs(exact_h_diagonals - lamda * adahess_diag_1.data)
                        .sum()
                        .item()
                    )
                elif method == "BL89":
                    summed_errors[method][name] = (
                        torch.abs(exact_h_diagonals - lamda * param_soft.hesscale.data)
                        .sum()
                        .item()
                    )
                else:
                    summed_errors[method][name] = (
                        torch.abs(
                            exact_h_diagonals
                            - lamda * getattr(param, methods[method]).data
                        )
                        .sum()
                        .item()
                    )

        return summed_errors

    def learn(self, state, label):
        label = torch.tensor(label).long()
        state = torch.tensor(state).float()

        self.optimizer.zero_grad()
        self.optimizer_softmax.zero_grad()

        loss = self.loss_func(self.model(state), label)
        loss_softmax = self.loss_func_log(self.model(state), label)

        loss.backward()
        loss_softmax.backward()

        self.optimizer.step()
        self.optimizer_softmax.step()

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

    def get_adahess_estimate(self, model_parameters, mc_samples=1):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        params = []
        grads = []
        for p in model_parameters:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)

        self.single_gpu = True

        # Check backward was called with create_graph set to True
        for i, grad in enumerate(grads):
            if grad.grad_fn is None:
                raise RuntimeError(
                    "Gradient tensor {:} does not have grad_fn. When calling\n".format(
                        i
                    )
                    + "\t\t\t  loss.backward(), make sure the option create_graph is\n"
                    + "\t\t\t  set to True."
                )

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
                    v_i[v_i < 0.0] = -1.0
                    v_i[v_i >= 0.0] = 1.0

            if hvs_sum is None:
                hvs_sum = [
                    x * (1 / mc_samples)
                    for x in torch.autograd.grad(
                        grads,
                        params,
                        grad_outputs=v,
                        only_inputs=True,
                        retain_graph=True,
                    )
                ]
            else:
                hvs_sum += [
                    x * (1 / mc_samples)
                    for x in torch.autograd.grad(
                        grads,
                        params,
                        grad_outputs=v,
                        only_inputs=True,
                        retain_graph=True,
                    )
                ]

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


def linear_counter(network):
    counter = 0
    for layer in network:
        if layer.__class__ == torch.nn.modules.linear.Linear:
            counter += 1
    return counter
