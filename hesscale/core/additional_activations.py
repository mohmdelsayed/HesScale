from torch import Tensor, exp
from torch.nn import Module

class Exponential(Module):
    r"""Applies the exponential function element-wise:

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self):
        super(Exponential, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return exp(input)

    def extra_repr(self) -> str:
        inplace_str = ''
        return inplace_str