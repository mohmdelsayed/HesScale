from backpack.core.derivatives import (
    CrossEntropyLossDerivatives,
    ELUDerivatives,
    LeakyReLUDerivatives,
    LinearDerivatives,
    LogSigmoidDerivatives,
    MSELossDerivatives,
    ReLUDerivatives,
    SELUDerivatives,
    SigmoidDerivatives,
    TanhDerivatives,
)
from backpack.core.derivatives.elementwise import ElementwiseDerivatives
from torch import einsum, ones, zeros, zeros_like
from torch.nn.functional import softmax


class LinearDerivativesHesScale(LinearDerivatives):
    def diag_hessian(self, module, g_inp, g_out, mat):
        if "L" in mat:
            return (
                einsum("oi,vno->vni", (module.weight.data ** 2, mat[0])),
                einsum("nd,di->ni", (g_out[0], module.weight.data)).unsqueeze(0),
            )
        else:
            return (
                einsum("oi,vno->vni", (module.weight.data ** 2, mat[0] + mat[1])),
                einsum("nd,di->...ni", (g_out[0], module.weight.data)),
            )


class MSELossDerivativesHesScale(MSELossDerivatives):
    def diag_hessian(self, module, g_inp, g_out):
        diag_H = ones(1, module.input0.size(0), module.input0.size(1)) * 2.0
        if module.reduction == "mean":
            diag_H /= module.input0.numel()
        return (diag_H, "L")


class CrossEntropyLossDerivativesHesScale(CrossEntropyLossDerivatives):
    def diag_hessian(self, module, g_inp, g_out):
        self._check_2nd_order_parameters(module)
        probs = self._get_probs(module)
        diag_H = (probs - probs ** 2).unsqueeze_(0)
        if module.reduction == "mean":
            diag_H /= module.input0.shape[0]
        return (diag_H, "L")


class NLLLossDerivativesHesScale:
    def diag_hessian(self, module, g_inp, g_out):
        diag_H = zeros(1, module.input0.size(0), module.input0.size(1))
        return (diag_H, "L")


class BaseActivationDerivatives(ElementwiseDerivatives):
    def __init__(self):
        super().__init__()

    def diag_hessian(self, module, g_inp, g_out, mat):
        self._no_inplace(module)

        if "L" in mat:
            return (
                einsum("mn,...mn->...mn", (self.df(module, g_inp, g_out) ** 2, mat[0])),
                einsum("mn,mn->...mn", (g_out[0], self.d2f(module, g_inp, g_out))),
            )
        else:
            return (
                einsum(
                    "mn, ...mn->...mn", (self.df(module, g_inp, g_out) ** 2, mat[0])
                ),
                einsum("mn, ...mn->...mn", (self.d2f(module, g_inp, g_out), mat[1])),
            )


class TanhDerivativesHesScale(BaseActivationDerivatives, TanhDerivatives):
    def __init__(self):
        super().__init__()


class ReLUDerivativesHesScale(BaseActivationDerivatives, ReLUDerivatives):
    def __init__(self):
        super().__init__()

    def d2f(self, module, g_inp, g_out):
        return zeros_like(module.input0)


class LeakyReLUDerivativesHesScale(BaseActivationDerivatives, LeakyReLUDerivatives):
    def __init__(self):
        super().__init__()

    def d2f(self, module, g_inp, g_out):
        return zeros_like(module.input0)


class SELUDerivativesHesScale(BaseActivationDerivatives, SELUDerivatives):
    def __init__(self):
        super().__init__()


class ELUDerivativesHesScale(BaseActivationDerivatives, ELUDerivatives):
    def __init__(self):
        super().__init__()


class SigmoidDerivativesHesScale(BaseActivationDerivatives, SigmoidDerivatives):
    def __init__(self):
        super().__init__()


class LogSigmoidDerivativesHesScale(BaseActivationDerivatives, LogSigmoidDerivatives):
    def __init__(self):
        super().__init__()


class LogSoftmaxDerivativesHesScale(BaseActivationDerivatives):
    def __init__(self):
        super().__init__()

    def hessian_is_zero(self):
        return False

    def df(self, module, g_inp, g_out):
        probs = self._get_probs(module)
        return 1 - probs

    def d2f(self, module, g_inp, g_out):
        probs = self._get_probs(module)
        return probs ** 2 - probs

    def _get_probs(self, module):
        return softmax(module.input0, dim=1)


class SoftmaxDerivativesHesScale(BaseActivationDerivatives):
    def __init__(self):
        super().__init__()

    def hessian_is_zero(self):
        return False

    def df(self, module, g_inp, g_out):
        probs = self._get_probs(module)
        return probs - probs ** 2

    def d2f(self, module, g_inp, g_out):
        probs = self._get_probs(module)
        return probs - 3 * probs ** 2 + 2 * probs ** 3

    def _get_probs(self, module):
        return softmax(module.input0, dim=1)
