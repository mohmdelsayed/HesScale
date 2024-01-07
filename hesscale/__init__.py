from backpack.extensions.backprop_extension import BackpropExtension
from torch.nn import (
    ELU,
    SELU,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    CrossEntropyLoss,
    NLLLoss,
    Dropout,
    Flatten,
    LeakyReLU,
    Linear,
    LogSigmoid,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    MSELoss,
    ReLU,
    Sigmoid,
    Tanh,
    ZeroPad2d,
    LogSoftmax,
    Softmax,
    Softplus,
)
from .core.additional_losses import SoftmaxNLLLoss, GaussianNLLLossMu, GaussianNLLLossVar
from .core.additional_activations import Exponential

from .core import (
    activations,
    activations_gn,
    conv1d,
    conv1d_gn,
    conv2d,
    conv2d_gn,
    conv3d,
    conv3d_gn,
    convtranspose1d,
    convtranspose1d_gn,
    convtranspose2d,
    convtranspose2d_gn,
    convtranspose3d,
    convtranspose3d_gn,
    dropout,
    dropout_gn,
    flatten,
    flatten_gn,
    linear,
    linear_gn,
    losses,
    losses_gn,
    pooling,
    pooling_gn,
)


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
                MaxPool1d: pooling.HesScaleMaxPool1d(),
                AvgPool1d: pooling.HesScaleAvgPool1d(),
                MaxPool2d: pooling.HesScaleMaxPool2d(),
                AvgPool2d: pooling.HesScaleAvgPool2d(),
                MaxPool3d: pooling.HesScaleMaxPool3d(),
                AvgPool3d: pooling.HesScaleAvgPool3d(),
                Conv1d: conv1d.HesScaleConv1d(),
                Conv2d: conv2d.HesScaleConv2d(),
                Conv3d: conv3d.HesScaleConv3d(),
                ConvTranspose1d: convtranspose1d.HesScaleConvTranspose1d(),
                ConvTranspose2d: convtranspose2d.HesScaleConvTranspose2d(),
                ConvTranspose3d: convtranspose3d.HesScaleConvTranspose3d(),
                Dropout: dropout.HesScaleDropout(),
                Flatten: flatten.HesScaleFlatten(),
                ReLU: activations.ReLUHesScale(),
                Sigmoid: activations.SigmoidHesScale(),
                Tanh: activations.TanhHesScale(),
                LeakyReLU: activations.LeakyReLUHesScale(),
                LogSigmoid: activations.LogSigmoidHesScale(),
                ELU: activations.ELUHesScale(),
                SELU: activations.SELUHesScale(),
                LogSoftmax: activations.LogSoftmaxHesScale(),
                Softmax: activations.SoftmaxHesScale(),
                SoftmaxNLLLoss: losses.SoftmaxNLLLossHesScale(),
                GaussianNLLLossMu: losses.GaussianNLLLossMuHesScale(),
                GaussianNLLLossVar: losses.GaussianNLLLossVarHesScale(),
                Exponential: activations.ExponentialHesScale(),
                Softplus: activations.SoftPlusHesScale(),
            },
        )


class HesScaleGN(BackpropExtension):
    def __init__(self, savefield="hesscale_gn"):
        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses_gn.MSELossHesScale(),
                NLLLoss: losses_gn.NLLLossHesScale(),
                CrossEntropyLoss: losses_gn.CrossEntropyLossHesScale(),
                Linear: linear_gn.LinearHesScale(),
                MaxPool1d: pooling_gn.HesScaleMaxPool1d(),
                AvgPool1d: pooling_gn.HesScaleAvgPool1d(),
                MaxPool2d: pooling_gn.HesScaleMaxPool2d(),
                AvgPool2d: pooling_gn.HesScaleAvgPool2d(),
                MaxPool3d: pooling_gn.HesScaleMaxPool3d(),
                AvgPool3d: pooling_gn.HesScaleAvgPool3d(),
                Conv1d: conv1d_gn.HesScaleConv1d(),
                Conv2d: conv2d_gn.HesScaleConv2d(),
                Conv3d: conv3d_gn.HesScaleConv3d(),
                ConvTranspose1d: convtranspose1d_gn.HesScaleConvTranspose1d(),
                ConvTranspose2d: convtranspose2d_gn.HesScaleConvTranspose2d(),
                ConvTranspose3d: convtranspose3d_gn.HesScaleConvTranspose3d(),
                Dropout: dropout_gn.HesScaleDropout(),
                Flatten: flatten_gn.HesScaleFlatten(),
                ReLU: activations_gn.ReLUHesScale(),
                Sigmoid: activations_gn.SigmoidHesScale(),
                Tanh: activations_gn.TanhHesScale(),
                LeakyReLU: activations_gn.LeakyReLUHesScale(),
                LogSigmoid: activations_gn.LogSigmoidHesScale(),
                ELU: activations_gn.ELUHesScale(),
                SELU: activations_gn.SELUHesScale(),
                LogSoftmax: activations_gn.LogSoftmaxHesScale(),
                Softmax: activations_gn.SoftmaxHesScale(),
                SoftmaxNLLLoss: losses_gn.SoftmaxNLLLossHesScale(),
                GaussianNLLLossMu: losses_gn.GaussianNLLLossMuHesScale(),
                GaussianNLLLossVar: losses_gn.GaussianNLLLossVarHesScale(),
                Exponential: activations_gn.ExponentialHesScale(),
                Softplus: activations_gn.SoftPlusHesScale(),
            },
        )
