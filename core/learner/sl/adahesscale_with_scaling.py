from core.learner.sl.learner import Learner
from core.optim.adahesscale_with_scaling import AdaHesScaleScaled, AdaHesScaleSqrtScaled, AdaHesScaleAdamStyleScaled

class AdaHesScaleScaledLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHesScaleScaled
        name = "adahesscale_scaled"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class AdaHesScaleSqrtScaledLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHesScaleSqrtScaled
        name = "adahesscale_sqrt_scaled"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class AdaHesScaleAdamStyleScaledLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHesScaleAdamStyleScaled
        name = "adahesscale_adamstyle_scaled"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)
