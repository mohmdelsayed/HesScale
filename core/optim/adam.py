from torch.optim.optimizer import Optimizer
import torch, math


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.15,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps value: {}".format(eps))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(beta2))

        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay
        )

        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        self.zero_grad()
        loss, output = closure()
        loss.backward()
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of gradient^2 values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                beta1, beta2 = group["beta1"], group["beta2"]

                state["step"] += 1
                # perform weight decay
                p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad.data, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(
                    p.grad.data ** 2, alpha=1 - beta2
                )
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                denom = ((exp_avg_sq.sqrt()) / math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )

                step_size = group["lr"] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss, output
