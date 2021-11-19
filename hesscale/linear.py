from torch import einsum

from .derivatives import LinearDerivativesHesScale
from .hesscale_base import BaseModuleHesScale

import hesscale.utils.linear as LinUtils


class LinearHesScale(BaseModuleHesScale):
    def __init__(self):
        super().__init__(
            derivatives=LinearDerivativesHesScale(), params=["bias", "weight"]
        )

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        return LinUtils.extract_bias_diagonal(module, backproped, sum_batch=True)

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        return LinUtils.extract_weight_diagonal(module, backproped, sum_batch=True)
