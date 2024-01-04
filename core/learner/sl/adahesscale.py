from core.learner.sl.learner import Learner
from core.optim.adahesscale import AdaHesScale, AdaHesScaleSqrt, AdaHesScaleAdamStyle

class AdaHesScaleLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHesScale
        name = "adahesscale"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)


class AdaHesScaleSqrtLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHesScaleSqrt
        name = "adahesscale_sqrt"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class AdaHesScaleAdamStyleLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHesScaleAdamStyle
        name = "adahesscale_adamstyle"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)
