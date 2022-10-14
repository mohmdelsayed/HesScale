from hesscale.core.derivatives_ggn import Conv3DDerivativesHesScale
from hesscale.core.convnd_ggn import HesScaleConvND


class HesScaleConv3d(HesScaleConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv3DDerivativesHesScale(), params=["bias", "weight"]
        )
