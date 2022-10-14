from hesscale.derivatives_ggn import ConvTranspose1DDerivativesHesScale
from hesscale.convtransposend_ggn import HesScaleConvTransposeND


class HesScaleConvTranspose1d(HesScaleConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivativesHesScale(), params=["bias", "weight"]
        )
