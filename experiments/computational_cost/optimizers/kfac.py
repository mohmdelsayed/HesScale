import math
from torch import einsum, symeig, tensor

import torch
from backpack.extensions import KFAC
from torch import abs
from torch import max as tor_max
from torch import zeros_like
from torch.optim import Optimizer
from .utils import multiply_vec_with_kron_facs

NUMERICAL_STABILITY_CONSTANT = 1e-12


class KFACOptimizer(Optimizer):
    method = KFAC()

    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1.0):
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
        super(KFACOptimizer, self).__init__(params, defaults)

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

                curv_p = getattr(p, group["method_field"])

                kfac2, kfac1 = curv_p

                state["step"] += 1
                
                hess_param = torch.matmul(torch.diagonal(kfac2, 0).unsqueeze(1), torch.diagonal(kfac1, 0).unsqueeze(0))

                exp_avg.mul_(beta1).add_(p.grad.detach_(), alpha=1 - beta1)
                exp_hessian_diag.mul_(beta2).add_(hess_param, alpha=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                denom = exp_hessian_diag.add_(group["eps"])

                step_size = bias_correction2 * group["lr"] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)


                # # Tikhonov
                # pi = self.__compute_tikhonov_factor(kfac1, kfac2)
                # # shift for factor 1: pi * sqrt(gamma  + eta) = pi * sqrt(gamma)
                # shift1 = pi * math.sqrt(group["eps"])
                # # factor 2: 1 / pi * sqrt(gamma  + eta) = 1 / pi * sqrt(gamma)
                # shift2 = 1.0 / pi * math.sqrt(group["eps"])

                # # invert, take into account the diagonal term
                # inv_kfac1 = self.__inverse(kfac1, shift=shift1)
                # inv_kfac2 = self.__inverse(kfac2, shift=shift2)

                # grad_p_flat = p.grad.view(-1)
                # curv_adapted_grad = multiply_vec_with_kron_facs(
                #     [inv_kfac1, inv_kfac2], grad_p_flat
                # )
                # curv_adapted_grad = curv_adapted_grad.view_as(p.grad)

                # p.data.add_(curv_adapted_grad, alpha=-group["lr"])

    def __compute_tikhonov_factor(self, kfac1, kfac2):
        """Scalar pi from trace norm for factored Tikhonov regularization.

        For details, see Section 6.3 of the KFAC paper.

        TODO: Allow for choices other than trace norm.
        """
        (dim1, _), (dim2, _) = kfac1.shape, kfac2.shape
        trace1, trace2 = kfac1.trace(), kfac2.trace()
        pi_squared = (trace1 / dim1) / (trace2 / dim2)
        return pi_squared.sqrt()

    def __inverse(self, sym_mat, shift):
        """Invert sym_mat + shift * I"""
        eigvals, eigvecs = self.__eigen(sym_mat)
        # account for diagonal term added to the matrix
        eigvals.add_(shift)
        return self.__inv_from_eigen(eigvals, eigvecs)

    def __eigen(self, sym_mat):
        """Return eigenvalues and eigenvectors from eigendecomposition."""
        eigvals, eigvecs = symeig(torch.rand_like(sym_mat), eigenvectors=True)
        return eigvals, eigvecs

    def __inv_from_eigen(self, eigvals, eigvecs, truncate=NUMERICAL_STABILITY_CONSTANT):
        inv_eigvals = 1.0 / eigvals
        inv_eigvals.clamp_(min=0.0, max=1.0 / truncate)
        # return inv_eigvals, eigvecs
        return einsum("ij,j,kj->ik", (eigvecs, inv_eigvals, eigvecs))
