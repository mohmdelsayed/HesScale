from hesscale.core.hesscale_base import BaseModuleHesScale
from hesscale.core.derivatives import (
    AvgPool1DDerivativesHesScale,
    AvgPool2DDerivativesHesScale,
    AvgPool3DDerivativesHesScale,
    MaxPool1DDerivativesHesScale,
    MaxPool2DDerivativesHesScale,
    MaxPool3DDerivativesHesScale,
)


class HesScaleMaxPool1d(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=MaxPool1DDerivativesHesScale())


class HesScaleMaxPool2d(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=MaxPool2DDerivativesHesScale())


class HesScaleMaxPool3d(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=MaxPool3DDerivativesHesScale())


class HesScaleAvgPool1d(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=AvgPool1DDerivativesHesScale())


class HesScaleAvgPool2d(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=AvgPool2DDerivativesHesScale())


class HesScaleAvgPool3d(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=AvgPool3DDerivativesHesScale())
