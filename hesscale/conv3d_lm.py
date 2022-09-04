from hesscale.derivatives_lm import Conv3DDerivativesHesScale
from hesscale.convnd_lm import HesScaleConvND


class HesScaleConv3d(HesScaleConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv3DDerivativesHesScale(), params=["bias", "weight"]
        )
