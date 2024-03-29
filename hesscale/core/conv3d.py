from hesscale.core.derivatives import Conv3DDerivativesHesScale
from hesscale.core.convnd import HesScaleConvND


class HesScaleConv3d(HesScaleConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv3DDerivativesHesScale(), params=["bias", "weight"]
        )
