from hesscale.core.hesscale_base import BaseModuleHesScale
from hesscale.utils import conv_ggn as convUtils


class HesScaleConvND(BaseModuleHesScale):
    def bias(self, ext, module, grad_inp, grad_out, backproped):
        return convUtils.extract_bias_diagonal(module, backproped, sum_batch=True)

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        weight_diag = convUtils.extract_weight_diagonal(
            module, backproped, sum_batch=True
        )
        return weight_diag
