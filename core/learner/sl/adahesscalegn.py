from core.learner.sl.learner import Learner
from core.optim.adahesscalegn import AdaHesScaleGN

class AdaHesScaleGNLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHesScaleGN
        name = "adahesscalegn"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)
