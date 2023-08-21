from core.learner.learner import Learner
from core.optim.adahesscale import AdaHesScale

class AdaHesScaleLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHesScale
        name = "adahesscale"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)
