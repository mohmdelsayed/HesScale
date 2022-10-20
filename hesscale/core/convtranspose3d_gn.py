from hesscale.core.derivatives_gn import ConvTranspose3DDerivativesHesScale
from hesscale.core.convtransposend_gn import HesScaleConvTransposeND


class HesScaleConvTranspose3d(HesScaleConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivativesHesScale(), params=["bias", "weight"]
        )
