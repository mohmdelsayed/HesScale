from torch import einsum

from .derivatives import LinearDerivativesHesScale
from .hesscale_base import BaseModuleHesScale


class LinearHesScale(BaseModuleHesScale):
    def __init__(self):
        super().__init__(
            derivatives=LinearDerivativesHesScale(), params=["bias", "weight"]
        )

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        if "L" in backproped:
            return einsum("vno->o", (backproped[0]))
        else:
            return einsum("vno->o", (backproped[0] + backproped[1]))

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        if "L" in backproped:
            return einsum("vno,ni->oi", (backproped[0], module.input0 ** 2))
        else:
            return einsum(
                "vno,ni->oi", (backproped[0] + backproped[1], module.input0 ** 2)
            )
