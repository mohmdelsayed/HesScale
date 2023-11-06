import torch, sys
from core.utils import tasks, networks, learners, criterions
from core.network.fcn_relu import FCNReLUSmallSoftmax
from core.network.fcn_leakyrelu import FCNLeakyReLUSmallSoftmax
from core.network.fcn_tanh import FCNTanhSmallSoftMax
from core.logger import Logger
from backpack import extend
import torch.nn as nn
import signal
import traceback
import time
import copy
from functools import partial
from core.utils import get_kfac_estimate, get_adahess_estimate

from backpack import backpack, extend
from backpack.extensions import DiagGGNExact, DiagGGNMC, DiagHessian, KFAC
from hesscale import HesScale, HesScaleGN
DiagGGNMC_1 = DiagGGNMC(mc_samples=1)
DiagGGNMC_1.savefield = "diag_ggn_mc_1"
DiagGGNMC_50 = DiagGGNMC(mc_samples=50)
DiagGGNMC_50.savefield = "diag_ggn_mc_50"
KFAC_1 = KFAC(mc_samples=1)
KFAC_1.savefield = "kfac_1"
KFAC_50 = KFAC(mc_samples=50)
KFAC_50.savefield = "kfac_50"

def signal_handler(msg, signal, frame):
    print('Exit signal: ', signal)
    cmd, learner = msg
    with open(f'timeout_{learner}.txt', 'a') as f:
        f.write(f"{cmd} \n")
    exit(0)

class RunApproxQuality:
    def __init__(self, name='approx_quality', n_samples=10000, task='stationary_mnist', exp_name='exp1', learner='sgd', save_path="logs", seed=0, network='fcn_relu', n_eval_samples=10, **kwargs):
        self.name = name
        self.n_samples = int(n_samples)
        self.n_eval_samples = int(n_eval_samples)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task_batch_size_32 = tasks[task](batch_size=32)
        self.task_batch_size_1 = tasks[task](batch_size=1)

        self.exp_name = exp_name
        self.learner = learners[learner](networks[network], kwargs)
        self.logger = Logger(save_path)
        self.seed = int(seed)
        
        self.methods = {
            "HS": "hesscale",
            "BL89": None,
            "AdaHess_1": None,
            "AdaHess_50": None,

            "GGN": "diag_ggn_exact",
            "HSGGN": "hesscale_gn",
            "GGNMC_1": "diag_ggn_mc_1",
            "GGNMC_50": "diag_ggn_mc_50",
            "KFAC_1": None,
            "KFAC_50": None,
            "g2": None,
            "H": "diag_h",
            "|H|": None,
        }

    def start(self):
        torch.manual_seed(self.seed)
        losses_per_step_size = []

        if self.task_batch_size_32.criterion == 'cross_entropy':
            accuracy_per_step_size = []
        
        self.learner.extend = True
        self.learner.setup_task(self.task_batch_size_32)
        criterion = extend(criterions[self.task_batch_size_32.criterion]()) if self.learner.extend else criterions[self.task_batch_size_32.criterion]()

        self.learner_NLL = copy.deepcopy(self.learner)
        self.learner_NLL.network_cls = FCNReLUSmallSoftmax
        self.learner_NLL.extend = True
        self.learner_NLL.setup_task(self.task_batch_size_32)
        criterion_NLL = nn.NLLLoss(reduction="mean")
        criterion_NLL = extend(criterion_NLL)

        l1_errors_before = self.compute_l1_errors(criterion, criterion_NLL)

        # training
        for _ in range(self.n_samples):
            input, target = next(self.task_batch_size_32)
            input, target = input.to(self.device), target.to(self.device)
            def closure_learner():
                output = self.learner.predict(input)
                loss = criterion(output, target)
                return loss, output
            def closure_learner_NLL():
                output = self.learner_NLL.predict(input)
                loss = criterion_NLL(output, target)
                return loss, output
            loss, output = self.learner.update_params(closure=closure_learner)
            _, _ = self.learner_NLL.update_params(closure=closure_learner_NLL)
            losses_per_step_size.append(loss.item())
            if self.task_batch_size_32.criterion == 'cross_entropy':
                accuracy_per_step_size.append((output.argmax(dim=1) == target).float().mean().item())

        l1_errors_after = self.compute_l1_errors(criterion, criterion_NLL)
        
        logging_data = {
                'losses': losses_per_step_size,
                'exp_name': self.exp_name,
                'task': self.task_batch_size_32.name,
                'learner': self.learner.name,
                'network': self.learner.network.name,
                'optimizer_hps': self.learner.optim_kwargs,
                'n_samples': self.n_samples,
                'seed': self.seed,
                'l1_errors_before': l1_errors_before,
                'l1_errors_after': l1_errors_after,
        }

        if self.task_batch_size_32.criterion == 'cross_entropy':
            logging_data['accuracies'] = accuracy_per_step_size

        self.logger.log(**logging_data)


    def compute_l1_errors(self, criterion, criterion_NLL):
        avgs_lists = {}
        for _ in range(self.n_eval_samples):
            input, target = next(self.task_batch_size_1)
            input, target = input.to(self.device), target.to(self.device)
            sample_error_per_method = self.compare_hess_methods(input, target, criterion, criterion_NLL)
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
    
    def __str__(self) -> str:
        return self.name

    def compare_hess_methods(self, input, target, criterion, criterion_NLL):

        output, output_NLL = self.learner.predict(input), self.learner_NLL.predict(input)
        loss, loss_NLL = criterion(output, target), criterion_NLL(output_NLL, target)
    
        self.learner.zero_grad()

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
            self.learner.parameters, mc_samples=1
        )

        adahess_diags_50 = get_adahess_estimate(
            self.learner.parameters, mc_samples=50
        )

        self.learner_NLL.zero_grad()
        
        with backpack(HesScale()):
            loss_NLL.backward()

        summed_errors = {}

        for method in self.methods:
            summed_errors[method] = {}

        for (name, param), (name_soft, param_soft), adahess_diag_1, adahess_diag_50 in zip(
            self.learner.named_parameters,
            self.learner_NLL.named_parameters,
            adahess_diags_1,
            adahess_diags_50
        ):  
            exact_h_diagonals = param.diag_h.data
            kfac_estimate_1 = get_kfac_estimate(param.kfac_1, param)
            kfac_estimate_50 = get_kfac_estimate(param.kfac_50, param)

            for method in self.methods:
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
                        torch.abs(exact_h_diagonals - getattr(param, self.methods[method]).data)
                        .sum()
                        .item()
                    )
        return summed_errors


if __name__ == "__main__":
    # start the run using the command line arguments
    ll = sys.argv[1:]
    args = {k[2:]:v for k,v in zip(ll[::2], ll[1::2])}
    run = RunApproxQuality(**args)
    cmd = f"python3 {' '.join(sys.argv)}"
    signal.signal(signal.SIGUSR1, partial(signal_handler, (cmd, args['learner'])))
    current_time = time.time()
    try:
        run.start()
        with open(f"finished_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} time_elapsed: {time.time()-current_time} \n")
    except Exception as e:
        with open(f"failed_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} \n")
        with open(f"failed_{args['learner']}_msgs.txt", "a") as f:
            f.write(f"{cmd} \n")
            f.write(f"{traceback.format_exc()} \n\n")