from hesscale.derivatives_lm import ConvTranspose3DDerivativesHesScale
from hesscale.convtransposend_lm import HesScaleConvTransposeND


class HesScaleConvTranspose3d(HesScaleConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivativesHesScale(), params=["bias", "weight"]
        )
