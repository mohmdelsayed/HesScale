from core.learner.sl.learner import Learner
from core.optim.adahg import AdaHG
from core.optim.adahg_sqrt import AdaHGSqrt

class AdaHGLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHG
        name = "adahg"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class AdaHGSqrtLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHGSqrt
        name = "adahg_sqrt"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)
