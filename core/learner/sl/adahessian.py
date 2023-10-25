from core.learner.sl.learner import Learner
from core.optim.adahessian import AdaHessian

class AdaHessianLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHessian
        name = "adahessian"
        super().__init__(name, network, optimizer, optim_kwargs)
