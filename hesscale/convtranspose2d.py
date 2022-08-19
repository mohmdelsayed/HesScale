from hesscale.derivatives import ConvTranspose2DDerivativesHesScale
from hesscale.convtransposend import HesScaleConvTransposeND


class HesScaleConvTranspose2d(HesScaleConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivativesHesScale(), params=["bias", "weight"]
        )
