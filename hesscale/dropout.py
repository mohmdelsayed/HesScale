from hesscale.derivatives import DropoutDerivativesHesScale
from hesscale.hesscale_base import BaseModuleHesScale


class HesScaleDropout(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=DropoutDerivativesHesScale())
