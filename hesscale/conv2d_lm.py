from hesscale.derivatives_lm import Conv2DDerivativesHesScale
from hesscale.convnd_lm import HesScaleConvND


class HesScaleConv2d(HesScaleConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DDerivativesHesScale(), params=["bias", "weight"]
        )