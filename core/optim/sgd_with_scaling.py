import torch, math

class SGDScaled(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta=0.999, delta=1e-8, max_scale=1.0):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta=beta, delta=delta, max_scale=max_scale)
        super(SGDScaled, self).__init__(params, defaults)

    def step(self, closure=None):
        self.zero_grad()
        loss, output = closure()
        loss.backward()
        trust_region_term = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"] += 1
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq.mul_(group["beta"]).addcmul_(p.grad, p.grad, value=1-group["beta"])
                state["u"] = group["lr"] * (p.grad + group['weight_decay'] * p.data)
                trust_region_term += (exp_avg_sq * (state["u"] ** 2) / (1- group["beta"] ** state["step"])).sum()
        
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                scaled_step_size = min(group["max_scale"], math.sqrt(2 * group["delta"] / trust_region_term))
                p.data.add_(state["u"], alpha=-scaled_step_size)
        return loss, output

import torch, math

class SGDScaledSqrt(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta=0.999, delta=1e-8, max_scale=1.0):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta=beta, delta=delta, max_scale=max_scale)
        super(SGDScaledSqrt, self).__init__(params, defaults)

    def step(self, closure=None):
        self.zero_grad()
        loss, output = closure()
        loss.backward()
        trust_region_term = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"] += 1
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq.mul_(group["beta"]).addcmul_(p.grad, p.grad, value=1-group["beta"])
                state["u"] = group["lr"] * (p.grad + group['weight_decay'] * p.data)
                trust_region_term += (exp_avg_sq.sqrt() * (state["u"] ** 2) / (1- group["beta"] ** state["step"])).sum()
        
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                scaled_step_size = min(group["max_scale"], math.sqrt(2 * group["delta"] / trust_region_term))
                p.data.add_(state["u"], alpha=-scaled_step_size)
        return loss, output