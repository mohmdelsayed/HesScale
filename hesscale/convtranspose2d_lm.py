from hesscale.derivatives_lm import ConvTranspose2DDerivativesHesScale
from hesscale.convtransposend_lm import HesScaleConvTransposeND


class HesScaleConvTranspose2d(HesScaleConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivativesHesScale(), params=["bias", "weight"]
        )
