from hesscale.derivatives_lm import Conv1DDerivativesHesScale
from hesscale.convnd_lm import HesScaleConvND


class HesScaleConv1d(HesScaleConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv1DDerivativesHesScale(), params=["bias", "weight"]
        )
