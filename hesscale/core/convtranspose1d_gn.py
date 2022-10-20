from hesscale.core.derivatives_gn import ConvTranspose1DDerivativesHesScale
from hesscale.core.convtransposend_gn import HesScaleConvTransposeND


class HesScaleConvTranspose1d(HesScaleConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivativesHesScale(), params=["bias", "weight"]
        )
