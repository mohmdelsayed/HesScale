import math

import torch
from hesscale import HesScaleLM
from torch import abs
from torch import max as tor_max
from torch import zeros_like
from torch.optim import Optimizer


class HesScaleLMOptimizer(Optimizer):
    method = HesScaleLM()

    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, method_field=type(self).method.savefield
        )
        super(HesScaleLMOptimizer, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = zeros_like(p.data)
                    # Exponential moving average of Hessian diagonal square values
                    state["exp_hessian_diag"] = zeros_like(p.data)

                exp_avg, exp_hessian_diag = state["exp_avg"], state["exp_hessian_diag"]

                beta1, beta2 = group["betas"]

                state["step"] += 1
                hess_param = getattr(p, group["method_field"]).detach()

                exp_avg.mul_(beta1).add_(p.grad.detach_(), alpha=1 - beta1)
                exp_hessian_diag.mul_(beta2).addcmul_(
                    hess_param, hess_param, value=1 - beta2
                )

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                denom = exp_hessian_diag.sqrt().add_(group["eps"])

                step_size = bias_correction1 * group["lr"] / math.sqrt(bias_correction2)

                p.data.addcdiv_(exp_avg, denom, value=-step_size)
