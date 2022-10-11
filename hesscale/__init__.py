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
)

from . import (
    activations,
    activations_ggn,
    conv1d,
    conv1d_ggn,
    conv2d,
    conv2d_ggn,
    conv3d,
    conv3d_ggn,
    convtranspose1d,
    convtranspose1d_ggn,
    convtranspose2d,
    convtranspose2d_ggn,
    convtranspose3d,
    convtranspose3d_ggn,
    dropout,
    dropout_ggn,
    flatten,
    flatten_ggn,
    linear,
    linear_ggn,
    losses,
    losses_ggn,
    pooling,
    pooling_ggn,
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
            },
        )


class HesScaleGGN(BackpropExtension):
    def __init__(self, savefield="hesscale_ggn"):
        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses_ggn.MSELossHesScale(),
                NLLLoss: losses_ggn.NLLLossHesScale(),
                CrossEntropyLoss: losses_ggn.CrossEntropyLossHesScale(),
                Linear: linear_ggn.LinearHesScale(),
                MaxPool1d: pooling_ggn.HesScaleMaxPool1d(),
                AvgPool1d: pooling_ggn.HesScaleAvgPool1d(),
                MaxPool2d: pooling_ggn.HesScaleMaxPool2d(),
                AvgPool2d: pooling_ggn.HesScaleAvgPool2d(),
                MaxPool3d: pooling_ggn.HesScaleMaxPool3d(),
                AvgPool3d: pooling_ggn.HesScaleAvgPool3d(),
                Conv1d: conv1d_ggn.HesScaleConv1d(),
                Conv2d: conv2d_ggn.HesScaleConv2d(),
                Conv3d: conv3d_ggn.HesScaleConv3d(),
                ConvTranspose1d: convtranspose1d_ggn.HesScaleConvTranspose1d(),
                ConvTranspose2d: convtranspose2d_ggn.HesScaleConvTranspose2d(),
                ConvTranspose3d: convtranspose3d_ggn.HesScaleConvTranspose3d(),
                Dropout: dropout_ggn.HesScaleDropout(),
                Flatten: flatten_ggn.HesScaleFlatten(),
                ReLU: activations_ggn.ReLUHesScale(),
                Sigmoid: activations_ggn.SigmoidHesScale(),
                Tanh: activations_ggn.TanhHesScale(),
                LeakyReLU: activations_ggn.LeakyReLUHesScale(),
                LogSigmoid: activations_ggn.LogSigmoidHesScale(),
                ELU: activations_ggn.ELUHesScale(),
                SELU: activations_ggn.SELUHesScale(),
                LogSoftmax: activations_ggn.LogSoftmaxHesScale(),
                Softmax: activations_ggn.SoftmaxHesScale(),
            },
        )
