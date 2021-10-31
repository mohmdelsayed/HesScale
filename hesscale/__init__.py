from backpack.extensions.backprop_extension import BackpropExtension
from torch.nn import (
    ELU,
    SELU,
    CrossEntropyLoss,
    LeakyReLU,
    Linear,
    LogSigmoid,
    LogSoftmax,
    MSELoss,
    NLLLoss,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
)

from . import activations, linear, losses


class HesScale(BackpropExtension):
    def __init__(self, savefield="hesscale"):
        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.MSELossHesScale(),
                NLLLoss: losses.NLLLossHesScale(),
                CrossEntropyLoss: losses.CrossEntropyLossHesScale(),
                Linear: linear.LinearHesScale(),
                ReLU: activations.ReLUHesScale(),
                Sigmoid: activations.SigmoidHesScale(),
                Tanh: activations.TanhHesScale(),
                LeakyReLU: activations.LeakyReLUHesScale(),
                LogSigmoid: activations.LogSigmoidHesScale(),
                ELU: activations.ELUHesScale(),
                SELU: activations.SELUHesScale(),
                LogSoftmax: activations.LogSoftmaxHesScale(),
                Softmax: activations.SoftmaxHesScale(),
            },
        )
