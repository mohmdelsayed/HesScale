import torch
from einops import rearrange
from torch import einsum
from torch.nn.functional import conv1d, conv2d, conv3d, unfold
from hesscale.derivatives import LOSS, ACTIVATION, LINEAR, CONV


def unfold_input(module, input):
    """Return unfolded input to a convolution.

    Use PyTorch's ``unfold`` operation for 2d convolutions (4d input tensors),
    otherwise fall back to a custom implementation.

    Args:
        module (torch.nn.Conv1d or torch.nn.Conv2d or torch.nn.Conv3d): Convolution
            module whose hyperparameters are used for the unfold.
        input (torch.Tensor): Input to convolution that will be unfolded.

    Returns:
        torch.Tensor: Unfolded input.
    """
    if input.dim() == 4:
        return unfold(
            input,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )
    else:
        return unfold_by_conv(input, module)


def get_weight_gradient_factors(input, grad_out, module, N):
    X = unfold_input(module, input)
    dE_dY = rearrange(grad_out, "n c ... -> n c (...)")
    return X, dE_dY


def separate_channels_and_pixels(module, tensor):
    """Reshape (V, N, C, H, W) into (V, N, C, H * W)."""
    return rearrange(tensor, "v n c ... -> v n c (...)")


def extract_weight_diagonal(module, backpropagated, sum_batch=True):
    if LOSS in backpropagated:

        unfolded_input = unfold_input(module, module.input0 ** 2)

        S = rearrange(backpropagated[0], "v n (g c) ... -> v n g c (...)", g=module.groups)
        unfolded_input = rearrange(unfolded_input, "n (g c) k -> n g c k", g=module.groups)

        JS = einsum("ngkl,vngml->vngmk", (unfolded_input, S))

        sum_dims = [0, 1] if sum_batch else [0]
        out_shape = (
            module.weight.shape if sum_batch else (JS.shape[1], *module.weight.shape)
        )

        weight_diagonal = JS.sum(sum_dims).reshape(out_shape)
        return weight_diagonal

    else:
        unfolded_input = unfold_input(module, module.input0 ** 2)

        S = rearrange(backpropagated[0]+backpropagated[1], "v n (g c) ... -> v n g c (...)", g=module.groups)
        unfolded_input = rearrange(unfolded_input, "n (g c) k -> n g c k", g=module.groups)

        JS = einsum("ngkl,vngml->vngmk", (unfolded_input, S))

        sum_dims = [0, 1] if sum_batch else [0]
        out_shape = (
            module.weight.shape if sum_batch else (JS.shape[1], *module.weight.shape)
        )

        weight_diagonal = JS.sum(sum_dims).reshape(out_shape)
        return weight_diagonal



def extract_bias_diagonal(module, backpropagated, sum_batch=True):

    if LOSS in backpropagated:
        start_spatial = 3
        sum_before = list(range(start_spatial, backpropagated[0].dim()))
        sum_after = [0, 1] if sum_batch else [0]

        return backpropagated[0].sum(sum_before).sum(sum_after)

    else:
        start_spatial = 3
        sum_before = list(range(start_spatial, backpropagated[0].dim()))
        sum_after = [0, 1] if sum_batch else [0]

        return (backpropagated[0]+backpropagated[1]).sum(sum_before).sum(sum_after)


def unfold_by_conv(input, module):
    """Return the unfolded input using convolution"""
    N, C_in = input.shape[0], input.shape[1]
    kernel_size = module.kernel_size
    kernel_size_numel = module.weight.shape[2:].numel()

    def make_weight():
        weight = torch.zeros(kernel_size_numel, 1, *kernel_size)

        for i in range(kernel_size_numel):
            extraction = torch.zeros(kernel_size_numel)
            extraction[i] = 1.0
            weight[i] = extraction.reshape(1, *kernel_size)

        repeat = [C_in, 1] + [1 for _ in kernel_size]
        return weight.repeat(*repeat)

    def get_conv():
        functional_for_module_cls = {
            torch.nn.Conv1d: conv1d,
            torch.nn.Conv2d: conv2d,
            torch.nn.Conv3d: conv3d,
        }
        return functional_for_module_cls[module.__class__]

    conv = get_conv()
    unfold = conv(
        input,
        make_weight().to(input.device),
        bias=None,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=C_in,
    )

    return unfold.reshape(N, C_in * kernel_size_numel, -1)
