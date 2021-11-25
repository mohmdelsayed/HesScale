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
    DropoutDerivatives,
)
from backpack.core.derivatives.elementwise import ElementwiseDerivatives
from torch import einsum, ones, zeros, zeros_like, cos, ones_like, sin, flip
from torch.nn.functional import softmax
from math import pi
from backpack.core.derivatives.conv_transposend import ConvTransposeNDDerivatives
from backpack.core.derivatives.convnd import ConvNDDerivatives
from backpack.core.derivatives.flatten import FlattenDerivatives

from backpack.core.derivatives.avgpoolnd import AvgPoolNDDerivatives
from backpack.core.derivatives.maxpoolnd import MaxPoolNDDerivatives
import torch
from einops import rearrange
from torch.nn.grad import _grad_input_padding

LOSS = "loss"
ACTIVATION = "activation"
LINEAR = "linear"
CONV = "conv"
FLATTEN = "flatten"

class ConvDerivativesHesScale(ConvNDDerivatives):
    def diag_hessian(self, module, g_inp, g_out, mat):
        if LOSS in mat:
            return (
                self.conv_matrix(module, mat[0].squeeze(0), sq=True).unsqueeze(0),
                self.conv_matrix(module, g_out[0], sq=False).unsqueeze(0),
                CONV,
            )
        elif ACTIVATION in mat:
            return (
                self.conv_matrix(
                    module, (mat[0] + mat[1]).squeeze(0), sq=True
                ).unsqueeze(0),
                self.conv_matrix(module, g_out[0], sq=False).unsqueeze(0),
                CONV,
            )
        elif CONV in mat or LINEAR in mat:
            raise NotImplementedError(
                "Consecetive Conv2d/Conv2d or Conv2d/Linear not supported"
            )
        else:
            raise NotImplementedError("Not supported conv")

    def conv_matrix(self, module, mat, sq=False):
        weight = module.weight ** 2 if sq else module.weight
        
        input_size = list(module.input0.size())
        input_size[0] = mat.size(0)

        grad_padding = _grad_input_padding(
            grad_output=mat,
            input_size=input_size,
            stride=module.stride,
            padding=module.padding,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
        )

        return self.conv_transpose_func(
            input=mat,
            weight=weight,
            bias=None,
            stride=module.stride,
            padding=module.padding,
            output_padding=grad_padding,
            groups=module.groups,
            dilation=module.dilation,
        )

class ConvTransposeDerivativesHesScale(ConvTransposeNDDerivatives):
    def diag_hessian(self, module, g_inp, g_out, mat):
        if LOSS in mat:
            return (
                self.conv_matrix(module, mat[0].squeeze(0), sq=True).unsqueeze(0),
                self.conv_matrix(module, g_out[0], sq=False).unsqueeze(0),
                CONV,
            )
        elif ACTIVATION in mat:
            return (
                self.conv_matrix(
                    module, (mat[0] + mat[1]).squeeze(0), sq=True
                ).unsqueeze(0),
                self.conv_matrix(module, g_out[0], sq=False).unsqueeze(0),
                CONV,
            )
        elif CONV in mat or LINEAR in mat:
            raise NotImplementedError(
                "Consecetive Conv2d/Conv2d or Conv2d/Linear not supported"
            )
        else:
            raise NotImplementedError("Not supported conv")

    def conv_matrix(self, module, mat, sq=False):
        jac_t = self.conv_func(
            mat,
            module.weight,
            bias=None,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

        for dim in range(self.conv_dims):
            axis = dim + 1
            size = module.input0.shape[axis]
            jac_t = jac_t.narrow(axis, 0, size)

        return jac_t


class AvgPoolNDDerivativesHesScale(AvgPoolNDDerivatives):
    def diag_hessian(self, module, g_inp, g_out, mat):
        if LOSS in mat:
            mat = mat[0]
        elif ACTIVATION in mat:
            mat = mat[0] + mat[1]
        elif CONV in mat or LINEAR in mat:
            raise NotImplementedError(
                "Consecetive Pooling/Conv2d or Pooling/Linear not supported"
            )
        else:
            raise NotImplementedError("Not supported conv")

        # self.check_exotic_parameters(module)
        # mat_as_pool = self.__make_single_channel(mat÷, module)
        jmp_as_pool = self.conv_matrix(module, mat)
        # self.__check_jmp_out_as_pool(mat, jmp_as_pool, module)

        return self.reshape_like_output(jmp_as_pool, module)
    def __make_single_channel(self, mat, module):
        """Create fake single-channel images, grouping batch,
        class and channel dimension."""
        result = rearrange(mat, "v n c ... -> (v n c) ...")
        C_axis = 1
        return result.unsqueeze(C_axis)
    
    def __check_jmp_out_as_pool(self, mat, jmp_as_pool, module):
        V = mat.size(0)
        if self.N == 1:
            N, C_out, L_out = module.output.shape
            assert jmp_as_pool.shape == (V * N * C_out, 1, L_out)
        elif self.N == 2:
            N, C_out, H_out, W_out = module.output.shape
            assert jmp_as_pool.shape == (V * N * C_out, 1, H_out, W_out)
        elif self.N == 3:
            N, C_out, D_out, H_out, W_out = module.output.shape
            assert jmp_as_pool.shape == (V * N * C_out, 1, D_out, H_out, W_out)

    def conv_matrix(self, module, mat):
        convnd = self.conv(
            in_channels=1,
            out_channels=1,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            bias=False,
        ).to(module.input0.device)

        convnd.weight.requires_grad = False
        avg_kernel = torch.ones_like(convnd.weight) / (convnd.weight.numel() ** 2)
        convnd.weight.data = avg_kernel

        return convnd(mat)


class MaxPoolNDDerivativesHesScale(MaxPoolNDDerivatives):
    def diag_hessian(self, module, g_inp, g_out, mat):
        if LOSS in mat:
            mat = mat[0]
        elif ACTIVATION in mat:
            mat = mat[0] + mat[1]
        elif CONV in mat or LINEAR in mat:
            raise NotImplementedError(
                "Consecetive Pooling/Conv2d or Pooling/Linear not supported"
            )
        else:
            raise NotImplementedError("Not supported conv")

        return self._jac_t_mat_prod(module, g_inp, g_out, mat)


class FlattenDerivativesHesScale(FlattenDerivatives):
    def __init__(self):
        super().__init__()

    def diag_hessian(self, module, g_inp, g_out, mat):
        if LINEAR in mat:
            return (
                self.reshape_like_input(mat[0], module),
                self.reshape_like_input(mat[1], module),
                LINEAR,
                FLATTEN,
            )
        elif ACTIVATION in mat:            
            return (
                self.reshape_like_input(mat[0], module),
                self.reshape_like_input(mat[1], module),
                ACTIVATION,
                FLATTEN,
            )
        elif LOSS in mat:
            return (
                self.reshape_like_input(mat[0], module),
                LOSS,
                FLATTEN,
            )
        else:
            raise NotImplemented("Not supported")


#############################################
#                   MLP                     #
#############################################


class LinearDerivativesHesScale(LinearDerivatives):
    def diag_hessian(self, module, g_inp, g_out, mat):
        if LOSS in mat:
            return (
                einsum("oi,vno->vni", (module.weight.data ** 2, mat[0])),
                einsum("nd,di->ni", (g_out[0], module.weight.data)).unsqueeze(0),
                LINEAR,
            )
        elif ACTIVATION in mat:
            return (
                einsum("oi,vno->vni", (module.weight.data ** 2, mat[0] + mat[1])),
                einsum("nd,di->...ni", (g_out[0], module.weight.data)),
                LINEAR,
            )
        elif LINEAR in mat: #TODO
            return (
                einsum("oi,vno->vni", (module.weight.data ** 2, mat[0])),
                einsum("nd,di->ni", (mat[1].squeeze(0), module.weight.data)).unsqueeze(0),
                LINEAR,
            )
        else:
            raise NotImplementedError("Not supported")


#############################################
#                 Losses                    #
#############################################


class MSELossDerivativesHesScale(MSELossDerivatives):
    def diag_hessian(self, module, g_inp, g_out):
        diag_H = ones(1, module.input0.size(0), module.input0.size(1)) * 2.0
        if module.reduction == "mean":
            diag_H /= module.input0.numel()
        return (diag_H, LOSS)


class CrossEntropyLossDerivativesHesScale(CrossEntropyLossDerivatives):
    def diag_hessian(self, module, g_inp, g_out):
        self._check_2nd_order_parameters(module)
        probs = self._get_probs(module)
        diag_H = (probs - probs ** 2).unsqueeze_(0)
        if module.reduction == "mean":
            diag_H /= module.input0.shape[0]
        return (diag_H, LOSS)


class RastriginLossDerivativesHesScale:
    def diag_hessian(self, module, g_inp, g_out):
        diag_H = zeros_like(module.input0)
        diag_H[0] = 2.0 + 40.0 * (pi ** 2) * cos(2 * pi * module.input0[0])
        diag_H[1] = 2.0 + 40.0 * (pi ** 2) * cos(2 * pi * module.input0[1])
        if module.reduction == "mean":
            diag_H /= module.input0.shape[0]
        return diag_H


class RosenbrockLossDerivativesHesScale:
    def diag_hessian(self, module, g_inp, g_out):
        diag_H = zeros_like(module.input0)
        diag_H[0] = 2.0 - 400.0 * module.input0[1] + 1200.0 * (module.input0[0] ** 2)
        diag_H[1] = 200.0
        if module.reduction == "mean":
            diag_H /= module.input0.shape[0]
        return diag_H


class NLLLossDerivativesHesScale:
    def diag_hessian(self, module, g_inp, g_out):
        diag_H = zeros(1, module.input0.size(0), module.input0.size(1))
        return (diag_H, LOSS)


#############################################
#              Activations                  #
#############################################


class BaseActivationDerivatives(ElementwiseDerivatives):
    def __init__(self):
        super().__init__()

    def diag_hessian(self, module, g_inp, g_out, mat):
        self._no_inplace(module)
        if len(g_inp[0].size()) == 4:
            equation = "mnop, ...mnop->...mnop"
        elif len(g_inp[0].size()) == 2:
            equation = "mn, ...mn->...mn"
        else:
            raise "Error: no valid dimension"

        if LOSS in mat:
            return (
                einsum(equation, (self.df(module, g_inp, g_out) ** 2, mat[0])),
                einsum(
                    equation, (g_out[0], self.d2f(module, g_inp, g_out).unsqueeze_(0))
                ),
                ACTIVATION,
            )
        elif LINEAR in mat or CONV in mat:
            return (
                einsum(equation, (self.df(module, g_inp, g_out) ** 2, mat[0])),
                einsum(equation, (self.d2f(module, g_inp, g_out), mat[1])),
                mat[1],
                self.d2f(module, g_inp, g_out),
                self.df(module, g_inp, g_out),
                0,
                ACTIVATION,
            )
        elif ACTIVATION in mat:
            if FLATTEN in mat:
                raise NotImplementedError(
                    "Flatten between two activations not supported"
                )
            prev_lin = mat[2]
            prev_d2f = mat[3]
            prev_df = mat[4]
            counter = mat[5]
            if counter > 0:
                raise NotImplementedError(
                    "Only two consecetive activations are supported"
                )

            return (
                einsum(equation, (self.df(module, g_inp, g_out) ** 2, mat[0])),
                einsum(
                    equation,
                    (
                        prev_d2f * self.df(module, g_inp, g_out) ** 2
                        + self.d2f(module, g_inp, g_out) * prev_df,
                        prev_lin,
                    ),
                ),
                mat[1],
                self.d2f(module, g_inp, g_out),
                self.df(module, g_inp, g_out),
                counter + 1,
                ACTIVATION,
            )

        else:
            raise NotImplementedError("Not supported layer")


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


class DropoutDerivativesHesScale(BaseActivationDerivatives, DropoutDerivatives):
    def __init__(self):
        super().__init__()

    def d2f(self, module, g_inp, g_out):
        return zeros_like(module.input0)


class TanhDerivativesHesScale(BaseActivationDerivatives, TanhDerivatives):
    def __init__(self):
        super().__init__()


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


class ConvTranspose1DDerivativesHesScale(ConvTransposeDerivativesHesScale):
    def __init__(self):
        super().__init__(N=1)


class ConvTranspose2DDerivativesHesScale(ConvTransposeDerivativesHesScale):
    def __init__(self):
        super().__init__(N=2)


class ConvTranspose3DDerivativesHesScale(ConvTransposeDerivativesHesScale):
    def __init__(self):
        super().__init__(N=3)


class Conv1DDerivativesHesScale(ConvDerivativesHesScale):
    def __init__(self):
        super().__init__(N=1)


class Conv2DDerivativesHesScale(ConvDerivativesHesScale):
    def __init__(self):
        super().__init__(N=2)


class Conv3DDerivativesHesScale(ConvDerivativesHesScale):
    def __init__(self):
        super().__init__(N=3)


class AvgPool1DDerivativesHesScale(AvgPoolNDDerivativesHesScale):
    def __init__(self):
        super().__init__(N=1)


class AvgPool2DDerivativesHesScale(AvgPoolNDDerivativesHesScale):
    def __init__(self):
        super().__init__(N=2)


class AvgPool3DDerivativesHesScale(AvgPoolNDDerivativesHesScale):
    def __init__(self):
        super().__init__(N=3)


class MaxPool1DDerivativesHesScale(MaxPoolNDDerivativesHesScale):
    def __init__(self):
        super().__init__(N=1)


class MaxPool2DDerivativesHesScale(MaxPoolNDDerivativesHesScale):
    def __init__(self):
        super().__init__(N=2)


class MaxPool3DDerivativesHesScale(MaxPoolNDDerivativesHesScale):
    def __init__(self):
        super().__init__(N=3)
