from .derivatives_gn import (
    CrossEntropyLossDerivativesHesScale,
    MSELossDerivativesHesScale,
    NLLLossDerivativesHesScale,
    GaussianNLLLossMuDerivativesHesScale,
    GaussianNLLLossVarDerivativesHesScale,
)
from .hesscale_base import BaseModuleHesScale


class LossHesScale(BaseModuleHesScale):
    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):
        return self.derivatives.diag_hessian(module, grad_inp, grad_out)


class MSELossHesScale(LossHesScale):
    def __init__(self):
        super().__init__(derivatives=MSELossDerivativesHesScale())


class CrossEntropyLossHesScale(LossHesScale):
    def __init__(self):
        super().__init__(derivatives=CrossEntropyLossDerivativesHesScale())


class NLLLossHesScale(LossHesScale):
    def __init__(self):
        super().__init__(derivatives=NLLLossDerivativesHesScale())

class SoftmaxNLLLossHesScale(LossHesScale):
    def __init__(self):
        super().__init__(derivatives=CrossEntropyLossDerivativesHesScale())

class GaussianNLLLossMuHesScale(LossHesScale):
    def __init__(self):
        super().__init__(derivatives=GaussianNLLLossMuDerivativesHesScale())

class GaussianNLLLossVarHesScale(LossHesScale):
    def __init__(self):
        super().__init__(derivatives=GaussianNLLLossVarDerivativesHesScale())