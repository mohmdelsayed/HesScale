from .derivatives_gn import (
    ELUDerivativesHesScale,
    LeakyReLUDerivativesHesScale,
    LogSigmoidDerivativesHesScale,
    LogSoftmaxDerivativesHesScale,
    ReLUDerivativesHesScale,
    SELUDerivativesHesScale,
    SigmoidDerivativesHesScale,
    SoftmaxDerivativesHesScale,
    TanhDerivativesHesScale,
)
from .hesscale_base import BaseModuleHesScale


class ReLUHesScale(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivativesHesScale())


class SigmoidHesScale(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=SigmoidDerivativesHesScale())


class TanhHesScale(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=TanhDerivativesHesScale())


class ELUHesScale(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=ELUDerivativesHesScale())


class SELUHesScale(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=SELUDerivativesHesScale())


class LeakyReLUHesScale(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=LeakyReLUDerivativesHesScale())


class LogSoftmaxHesScale(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=LogSoftmaxDerivativesHesScale())


class LogSigmoidHesScale(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=LogSigmoidDerivativesHesScale())


class SoftmaxHesScale(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=SoftmaxDerivativesHesScale())
