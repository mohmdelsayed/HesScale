from hesscale.core.derivatives_ggn import Conv1DDerivativesHesScale
from hesscale.core.convnd_ggn import HesScaleConvND


class HesScaleConv1d(HesScaleConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv1DDerivativesHesScale(), params=["bias", "weight"]
        )
