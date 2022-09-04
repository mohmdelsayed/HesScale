from .derivatives_lm import (
    CrossEntropyLossDerivativesHesScale,
    MSELossDerivativesHesScale,
    NLLLossDerivativesHesScale,
    RastriginLossDerivativesHesScale,
    RosenbrockLossDerivativesHesScale,
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


class RastriginLossHesScale(LossHesScale):
    def __init__(self):
        super().__init__(derivatives=RastriginLossDerivativesHesScale())


class RosenbrockLossHesScale(LossHesScale):
    def __init__(self):
        super().__init__(derivatives=RosenbrockLossDerivativesHesScale())
