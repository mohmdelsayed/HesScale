from hesscale.hesscale_base import BaseModuleHesScale
from hesscale.derivatives import FlattenDerivativesHesScale


class HesScaleFlatten(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=FlattenDerivativesHesScale())

    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):
        if self.derivatives.is_no_op(module):
            return backproped
        else:
            return super().backpropagate(ext, module, grad_inp, grad_out, backproped)
