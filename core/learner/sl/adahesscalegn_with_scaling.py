from core.learner.sl.learner import Learner
from core.optim.adahesscalegn_with_scaling import AdaHesScaleGNScaled, AdaHesScaleGNSqrtScaled, AdaHesScaleGNAdamStyleScaled

class AdaHesScaleGNScaledLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHesScaleGNScaled
        name = "adahesscalegn_scaled"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class AdaHesScaleGNSqrtScaledLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHesScaleGNSqrtScaled
        name = "adahesscalegn_sqrt_scaled"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class AdaHesScaleGNAdamStyleScaledLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHesScaleGNAdamStyleScaled
        name = "adahesscalegn_adamstyle_scaled"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)
