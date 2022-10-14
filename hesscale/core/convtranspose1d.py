from hesscale.core.derivatives import ConvTranspose1DDerivativesHesScale
from hesscale.core.convtransposend import HesScaleConvTransposeND


class HesScaleConvTranspose1d(HesScaleConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivativesHesScale(), params=["bias", "weight"]
        )
