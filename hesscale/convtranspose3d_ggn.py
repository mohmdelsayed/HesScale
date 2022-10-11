from hesscale.derivatives_ggn import ConvTranspose3DDerivativesHesScale
from hesscale.convtransposend_ggn import HesScaleConvTransposeND


class HesScaleConvTranspose3d(HesScaleConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivativesHesScale(), params=["bias", "weight"]
        )
