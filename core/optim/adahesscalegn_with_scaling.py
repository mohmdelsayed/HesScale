import math
from hesscale import HesScaleGN
from backpack import backpack
from torch import zeros_like
from torch.optim import Optimizer

class AdaHesScaleGNAdamStyleScaled(Optimizer):
    method = HesScaleGN()
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, max_scale=1, delta=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, method_field=type(self).method.savefield, max_scale=max_scale, delta=delta
        )
        super(AdaHesScaleGNAdamStyleScaled, self).__init__(params, defaults)

    def step(self, closure=None):
        trust_region_term = 0.0
        self.zero_grad()
        loss, output = closure()
        with backpack(type(self).method):
            loss.backward()
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = zeros_like(p.data)
                    # Exponential moving average of Hessian diagonal square values
                    state["exp_avg_sq"] = zeros_like(p.data)
                    # adahesscale update
                    state["u"] = zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                beta1, beta2 = group["betas"]

                state["step"] += 1
                hess_param = getattr(p, group["method_field"]).detach()

                exp_avg.mul_(beta1).add_(p.grad.detach_(), alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(
                    hess_param.data ** 2, alpha=1 - beta2
                )

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                denom = ((exp_avg_sq.sqrt()) / math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )
                step_size = group["lr"] / bias_correction1

                state["u"] = step_size * exp_avg / denom

                trust_region_term += (exp_avg_sq.sqrt() * (state["u"] ** 2)).sum()

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                scaled_step_size = min(group["max_scale"], math.sqrt(2 * group["delta"] / trust_region_term))
                p.data.add_(state["u"], alpha=-scaled_step_size)
        return loss, output
    

class AdaHesScaleGNSqrtScaled(Optimizer):
    method = HesScaleGN()
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, max_scale=1, delta=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, method_field=type(self).method.savefield, max_scale=max_scale, delta=delta
        )
        super(AdaHesScaleGNSqrtScaled, self).__init__(params, defaults)

    def step(self, closure=None):
        trust_region_term = 0.0
        self.zero_grad()
        loss, output = closure()
        with backpack(type(self).method):
            loss.backward()
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = zeros_like(p.data)
                    # Exponential moving average of Hessian diagonal square values
                    state["exp_avg_sq"] = zeros_like(p.data)
                    # adahesscale update
                    state["u"] = zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                beta1, beta2 = group["betas"]

                state["step"] += 1
                hess_param = getattr(p, group["method_field"]).detach()

                exp_avg.mul_(beta1).add_(p.grad.detach_(), alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(
                    hess_param.data.abs(), alpha=1 - beta2
                )
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                denom = ((exp_avg_sq.sqrt()) / math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )
                step_size = group["lr"] / bias_correction1
                
                state["u"] = step_size * exp_avg / denom

                trust_region_term += (exp_avg_sq.sqrt() * (state["u"] ** 2)).sum()

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                scaled_step_size = min(group["max_scale"], math.sqrt(2 * group["delta"] / trust_region_term))
                p.data.add_(state["u"], alpha=-scaled_step_size)
        return loss, output


class AdaHesScaleGNScaled(Optimizer):
    method = HesScaleGN()
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, max_scale=1, delta=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, method_field=type(self).method.savefield, max_scale=max_scale, delta=delta
        )
        super(AdaHesScaleGNScaled, self).__init__(params, defaults)

    def step(self, closure=None):
        trust_region_term = 0.0
        self.zero_grad()
        loss, output = closure()
        with backpack(type(self).method):
            loss.backward()
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = zeros_like(p.data)
                    # Exponential moving average of Hessian diagonal square values
                    state["exp_avg_sq"] = zeros_like(p.data)
                    # adahesscale update
                    state["u"] = zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                beta1, beta2 = group["betas"]

                state["step"] += 1
                hess_param = getattr(p, group["method_field"]).detach()

                exp_avg.mul_(beta1).add_(p.grad.detach_(), alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(
                    hess_param.data.abs(), alpha=1 - beta2
                )

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                denom = (exp_avg_sq / bias_correction2).add_(
                    group["eps"]
                )
                step_size = group["lr"] / bias_correction1

                state["u"] = step_size * exp_avg / denom

                trust_region_term += (exp_avg_sq * (state["u"] ** 2)).sum()

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                scaled_step_size = min(group["max_scale"], math.sqrt(2 * group["delta"] / trust_region_term))
                p.data.add_(state["u"], alpha=-scaled_step_size)
        return loss, output