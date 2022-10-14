from hesscale.core.derivatives_ggn import Conv2DDerivativesHesScale
from hesscale.core.convnd_ggn import HesScaleConvND


class HesScaleConv2d(HesScaleConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DDerivativesHesScale(), params=["bias", "weight"]
        )
