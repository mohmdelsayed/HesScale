"""Utility functions for extracting transpose convolution BackPACK quantities."""

import torch
from einops import rearrange
from torch import einsum
from torch.nn.functional import conv_transpose1d, conv_transpose2d, conv_transpose3d
from hesscale.utils.conv_ggn import unfold_input


def get_weight_gradient_factors(input, grad_out, module, N):
    M, C_in = input.shape[0], input.shape[1]
    kernel_size_numel = module.weight.shape[2:].numel()

    X = unfold_by_conv_transpose(input, module).reshape(M, C_in * kernel_size_numel, -1)
    dE_dY = rearrange(grad_out, "n c ... -> n c (...)")

    return X, dE_dY


def extract_weight_diagonal(module, backpropagated, sum_batch=True):

    unfolded_input = unfold_input(module, module.input0 ** 2)

    S = rearrange(backpropagated, "v n (g o) ... -> v n g o (...)", g=module.groups)
    unfolded_input = rearrange(
        unfolded_input,
        "n (g c) (k x) -> n g c k x",
        g=module.groups,
        k=module.weight.shape[2:].numel(),
    )

    JS = einsum("ngckx,vngox->vngcok", (unfolded_input, S))

    sum_dims = [0, 1] if sum_batch else [0]
    out_shape = (
        module.weight.shape if sum_batch else (JS.shape[1], *module.weight.shape)
    )

    weight_diagonal = JS.sum(sum_dims).reshape(out_shape)

    return weight_diagonal


def extract_bias_diagonal(module, backpropagated, sum_batch=True):
    start_spatial = 3
    sum_before = list(range(start_spatial, backpropagated.dim()))
    sum_after = [0, 1] if sum_batch else [0]
    return backpropagated.sum(sum_before).sum(sum_after)


def unfold_by_conv_transpose(input, module):
    """Return the unfolded input using one-hot transpose convolution.

    Args:
        input (torch.Tensor): Input to a transpose convolution.
        module (torch.nn.ConvTranspose1d or torch.nn.ConvTranspose2d or
            torch.nn.ConvTranspose3d): Transpose convolution layer that specifies
            the hyperparameters for unfolding.

    Returns:
        torch.Tensor: Unfolded input of shape ``(N, C, K * X)`` with
            ``K = module.weight.shape[2:].numel()`` the number of kernel elements
            and ``X = module.output.shape[2:].numel()`` the number of output pixels.
    """
    N, C_in = input.shape[0], input.shape[1]
    kernel_size = module.kernel_size
    kernel_size_numel = module.weight.shape[2:].numel()

    def make_weight():
        weight = torch.zeros(1, kernel_size_numel, *kernel_size)

        for i in range(kernel_size_numel):
            extraction = torch.zeros(kernel_size_numel)
            extraction[i] = 1.0
            weight[0, i] = extraction.reshape(*kernel_size)

        repeat = [C_in, 1] + [1 for _ in kernel_size]
        weight = weight.repeat(*repeat)
        return weight.to(module.weight.device)

    def get_conv_transpose():
        functional_for_module_cls = {
            torch.nn.ConvTranspose1d: conv_transpose1d,
            torch.nn.ConvTranspose2d: conv_transpose2d,
            torch.nn.ConvTranspose3d: conv_transpose3d,
        }
        return functional_for_module_cls[module.__class__]

    conv_transpose = get_conv_transpose()
    unfold = conv_transpose(
        input,
        make_weight().to(module.weight.device),
        bias=None,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=C_in,
    )

    return unfold.reshape(N, C_in, -1)
