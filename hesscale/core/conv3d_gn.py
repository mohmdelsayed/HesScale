from hesscale.core.derivatives_gn import Conv3DDerivativesHesScale
from hesscale.core.convnd_gn import HesScaleConvND


class HesScaleConv3d(HesScaleConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv3DDerivativesHesScale(), params=["bias", "weight"]
        )
