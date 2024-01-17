from core.learner.sl.learner import Learner
from core.optim.adahessian_with_scaling import AdaHessianScaled

class AdaHessianScaledLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHessianScaled
        name = "adahessian_scaled"
        super().__init__(name, network, optimizer, optim_kwargs)