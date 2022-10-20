from hesscale.core.derivatives_gn import Conv2DDerivativesHesScale
from hesscale.core.convnd_gn import HesScaleConvND


class HesScaleConv2d(HesScaleConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DDerivativesHesScale(), params=["bias", "weight"]
        )
