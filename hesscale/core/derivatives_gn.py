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
        return self.conv_matrix(module, mat.squeeze(0), sq=True).unsqueeze(0)

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
        return self.conv_matrix(module, mat.squeeze(0), sq=True).unsqueeze(0)

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
        self.check_exotic_parameters(module)
        mat_as_pool = self.__make_single_channel(mat, module)
        jmat_as_pool = self.__apply_jacobian_t_of(module, mat_as_pool)
        self.__check_jmp_in_as_pool(mat, jmat_as_pool, module)

        gout_as_pool = self.__make_single_channel(g_out[0].unsqueeze(0), module)
        jgout_as_pool = self.__apply_jacobian_t_of(module, gout_as_pool)
        self.__check_jmp_in_as_pool(g_out[0].unsqueeze(0), jgout_as_pool, module)

        return self.reshape_like_input(jmat_as_pool, module)

    def __make_single_channel(self, mat, module):
        """Create fake single-channel images, grouping batch,
        class and channel dimension."""
        result = rearrange(mat, "v n c ... -> (v n c) ...")
        C_axis = 1
        return result.unsqueeze(C_axis)

    def __apply_jacobian_t_of(self, module, mat):
        C_for_conv_t = 1

        convnd_t = self.convt(
            in_channels=C_for_conv_t,
            out_channels=C_for_conv_t,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            bias=False,
        ).to(module.input0.device)

        convnd_t.weight.requires_grad = False
        avg_kernel = torch.ones_like(convnd_t.weight) / convnd_t.weight.numel()
        convnd_t.weight.data = avg_kernel  # should be _pow(2)

        V_N_C_in = mat.size(0)
        if self.N == 1:
            _, _, L_in = module.input0.size()
            output_size = (V_N_C_in, C_for_conv_t, L_in)
        elif self.N == 2:
            _, _, H_in, W_in = module.input0.size()
            output_size = (V_N_C_in, C_for_conv_t, H_in, W_in)
        elif self.N == 3:
            _, _, D_in, H_in, W_in = module.input0.size()
            output_size = (V_N_C_in, C_for_conv_t, D_in, H_in, W_in)

        return convnd_t(mat, output_size=output_size)

    def __check_jmp_in_as_pool(self, mat, jmp_as_pool, module):
        V = mat.size(0)
        if self.N == 1:
            N, C_in, L_in = module.input0.size()
            assert jmp_as_pool.shape == (V * N * C_in, 1, L_in)
        elif self.N == 2:
            N, C_in, H_in, W_in = module.input0.size()
            assert jmp_as_pool.shape == (V * N * C_in, 1, H_in, W_in)
        elif self.N == 3:
            N, C_in, D_in, H_in, W_in = module.input0.size()
            assert jmp_as_pool.shape == (V * N * C_in, 1, D_in, H_in, W_in)


class MaxPoolNDDerivativesHesScale(MaxPoolNDDerivatives):
    def diag_hessian(self, module, g_inp, g_out, mat):

        mat_as_pool = rearrange(mat, "v n c ... -> v n c (...)")
        jmat_as_pool = self.__apply_jacobian_t_of(module, mat_as_pool)

        return self.reshape_like_input(jmat_as_pool, module)

    def __pool_idx_for_jac(self, module, V):
        """Manipulated pooling indices ready-to-use in jac(t)."""
        pool_idx = self.get_pooling_idx(module)
        pool_idx = rearrange(pool_idx, "n c ... -> n c (...)")

        V_axis = 0

        return pool_idx.unsqueeze(V_axis).expand(V, -1, -1, -1)

    def __apply_jacobian_t_of(self, module, mat):
        V = mat.shape[0]
        result = self.__zero_for_jac_t(module, V, mat.device)
        pool_idx = self.__pool_idx_for_jac(module, V)

        N_axis = 3
        result.scatter_add_(N_axis, pool_idx, mat)
        return result

    def __zero_for_jac_t(self, module, V, device):
        if self.N == 1:
            N, C_out, _ = module.output.shape
            _, _, L_in = module.input0.size()

            shape = (V, N, C_out, L_in)

        elif self.N == 2:
            N, C_out, _, _ = module.output.shape
            _, _, H_in, W_in = module.input0.size()

            shape = (V, N, C_out, H_in * W_in)

        elif self.N == 3:
            N, C_out, _, _, _ = module.output.shape
            _, _, D_in, H_in, W_in = module.input0.size()

            shape = (V, N, C_out, D_in * H_in * W_in)

        return zeros(shape, device=device)


class FlattenDerivativesHesScale(FlattenDerivatives):
    def __init__(self):
        super().__init__()

    def diag_hessian(self, module, g_inp, g_out, mat):
        return self.reshape_like_input(mat, module)


#############################################
#                   MLP                     #
#############################################


class LinearDerivativesHesScale(LinearDerivatives):
    def diag_hessian(self, module, g_inp, g_out, mat):
        return einsum("oi,vno->vni", (module.weight.data ** 2, mat))


#############################################
#                 Losses                    #
#############################################


class MSELossDerivativesHesScale(MSELossDerivatives):
    def diag_hessian(self, module, g_inp, g_out):
        diag_H = ones(1, module.input0.size(0), module.input0.size(1)) * 2.0
        if module.reduction == "mean":
            diag_H /= module.input0.numel()
        return diag_H


class CrossEntropyLossDerivativesHesScale(CrossEntropyLossDerivatives):
    def diag_hessian(self, module, g_inp, g_out):
        self._check_2nd_order_parameters(module)
        probs = self._get_probs(module)
        diag_H = (probs - probs ** 2).unsqueeze_(0)
        if module.reduction == "mean":
            diag_H /= module.input0.shape[0]
        return diag_H


class NLLLossDerivativesHesScale:
    def diag_hessian(self, module, g_inp, g_out):
        diag_H = zeros(1, module.input0.size(0), module.input0.size(1))
        return diag_H

class GaussianNLLLossMuDerivativesHesScale:
    def diag_hessian(self, module, g_inp, g_out):
        diag_H = -module.input3 / (module.input1 + module.eps)
        if module.reduction == "mean":
            diag_H /= module.input0.shape[0]
        return diag_H.unsqueeze_(0)

class GaussianNLLLossVarDerivativesHesScale:
    def diag_hessian(self, module, g_inp, g_out):
        diag_H = module.input3 * (0.5 -  ((module.input2 - module.input1) ** 2) / (module.input0 + module.eps) ) / (module.input0 ** 2 + module.eps)
        if module.reduction == "mean":
            diag_H /= module.input0.shape[0]
        return diag_H.unsqueeze_(0)
    
class SoftmaxPPOLossDerivativesHesScale:
    def diag_hessian(self, module, g_inp, g_out):
        new_prob = module.get_prob(module.input0, module.input3)
        old_prob = module.input1
        adv = module.input2
        ratio = new_prob / old_prob

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - module.epsilon, 1.0 + module.epsilon) * adv

        flag1 = (surr1 < surr2).float()
        flag2 = (ratio > 1-module.epsilon).float() * (ratio < 1+module.epsilon).float() 
        derivative = new_prob - new_prob ** 2
        hessian_diags = derivative - 2 * new_prob * derivative
        diag_H = (hessian_diags * adv / old_prob) * flag1 + (1-flag1) * flag2 * (hessian_diags * adv / old_prob)
        if module.reduction == "mean":
            diag_H /= module.input0.shape[0]
        return diag_H.unsqueeze_(0)

class GaussianNLLLossMuPPODerivativesHesScale:
    def diag_hessian(self, module, g_inp, g_out):
        probs = module.get_prob(module.input0, module.input1, module.input2)
        
        old_prob = module.input3
        adv = module.input4
        ratio = probs / old_prob
        action = module.input2
        mu = module.input0
        sigma = module.input1
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - module.epsilon, 1.0 + module.epsilon) * adv

        flag1 = (surr1 < surr2).float()
        flag2 = (ratio > 1-module.epsilon).float() * (ratio < 1+module.epsilon).float() 

        grad_mu = (action - mu) * probs / (sigma ** 2 + module.eps)
        diag_H = ((action - mu) * grad_mu - probs) / (sigma ** 2 + module.eps)
        diag_H = diag_H * flag1 * adv / old_prob + (1-flag1) * flag2 * diag_H * adv / old_prob

        if module.reduction == "mean":
            diag_H /= module.input0.shape[0]
        return diag_H.unsqueeze_(0)

class GaussianNLLLossVarPPODerivativesHesScale:
    def diag_hessian(self, module, g_inp, g_out):
        probs = module.get_prob(module.input1, module.input0, module.input2)
        
        old_prob = module.input3
        adv = module.input4
        ratio = probs / old_prob
        action = module.input2
        mu = module.input1
        sigma = module.input0
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - module.epsilon, 1.0 + module.epsilon) * adv

        flag1 = (surr1 < surr2).float()
        flag2 = (ratio > 1-module.epsilon).float() * (ratio < 1+module.epsilon).float() 

        grad_sigma_sq = 0.5 * probs * (((action - mu) ** 2) / (sigma ** 2 + module.eps) - 1.0) / (sigma ** 2 + module.eps)
        diag_H = 0.5 * (grad_sigma_sq * (sigma ** 2) - probs) * (((action - mu)**2) / (sigma ** 2 + module.eps) - 1.0) / (sigma ** 4 + module.eps) - 0.5 * probs * ((action - mu) ** 2) / (sigma ** 6 + module.eps)
        diag_H = diag_H * flag1 * adv / old_prob + (1-flag1) * flag2 * diag_H * adv / old_prob

        if module.reduction == "mean":
            diag_H /= module.input0.shape[0]
        return diag_H.unsqueeze_(0)

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

        return einsum(equation, (self.df(module, g_inp, g_out) ** 2, mat))


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

    def d2f(self, module, g_inp, g_out):
        return zeros_like(module.input0)


class SELUDerivativesHesScale(BaseActivationDerivatives, SELUDerivatives):
    def __init__(self):
        super().__init__()

    def d2f(self, module, g_inp, g_out):
        return zeros_like(module.input0)


class ELUDerivativesHesScale(BaseActivationDerivatives, ELUDerivatives):
    def __init__(self):
        super().__init__()

    def d2f(self, module, g_inp, g_out):
        return zeros_like(module.input0)


class SigmoidDerivativesHesScale(BaseActivationDerivatives, SigmoidDerivatives):
    def __init__(self):
        super().__init__()

    def d2f(self, module, g_inp, g_out):
        return zeros_like(module.input0)

class SoftPlusDerivativesHesScale(BaseActivationDerivatives):
    def __init__(self):
        super().__init__()

    def hessian_is_zero(self):
        return False

    def df(self, module, g_inp, g_out):
        return torch.sigmoid(module.input0)

    def d2f(self, module, g_inp, g_out):
        return torch.sigmoid(module.input0) * (1.0 - torch.sigmoid(module.input0))

class ExponentialDerivativesHesScale(BaseActivationDerivatives):
    def __init__(self):
        super().__init__()

    def hessian_is_zero(self):
        return False

    def df(self, module, g_inp, g_out):
        return module.output

    def d2f(self, module, g_inp, g_out):
        return zeros_like(module.input0)

class LogSigmoidDerivativesHesScale(BaseActivationDerivatives, LogSigmoidDerivatives):
    def __init__(self):
        super().__init__()

    def d2f(self, module, g_inp, g_out):
        return zeros_like(module.input0)


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
