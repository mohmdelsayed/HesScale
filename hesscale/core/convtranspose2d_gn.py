from hesscale.core.derivatives_gn import ConvTranspose2DDerivativesHesScale
from hesscale.core.convtransposend_gn import HesScaleConvTransposeND


class HesScaleConvTranspose2d(HesScaleConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivativesHesScale(), params=["bias", "weight"]
        )
