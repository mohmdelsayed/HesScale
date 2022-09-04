from hesscale.derivatives_lm import ConvTranspose1DDerivativesHesScale
from hesscale.convtransposend_lm import HesScaleConvTransposeND


class HesScaleConvTranspose1d(HesScaleConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivativesHesScale(), params=["bias", "weight"]
        )
