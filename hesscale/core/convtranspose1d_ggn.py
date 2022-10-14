from hesscale.core.derivatives_ggn import ConvTranspose1DDerivativesHesScale
from hesscale.core.convtransposend_ggn import HesScaleConvTransposeND


class HesScaleConvTranspose1d(HesScaleConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivativesHesScale(), params=["bias", "weight"]
        )
