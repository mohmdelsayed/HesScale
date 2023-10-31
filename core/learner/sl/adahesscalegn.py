from core.learner.sl.learner import Learner
from core.optim.adahesscalegn import AdaHesScaleGN, AdaHesScaleGNSqrt, AdaHesScaleGNAdamStyle

class AdaHesScaleGNLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHesScaleGN
        name = "adahesscalegn"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class AdaHesScaleGNSqrtLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHesScaleGNSqrt
        name = "adahesscalegn_sqrt"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class AdaHesScaleGNAdamStyleLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHesScaleGNAdamStyle
        name = "adahesscalegn_adamstyle"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)
