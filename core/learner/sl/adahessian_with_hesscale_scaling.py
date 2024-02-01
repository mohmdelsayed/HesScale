from core.learner.sl.learner import Learner
from core.optim.adahessian_with_hesscale_scaling import AdaHessianHesScaleScaled

class AdaHessianHesScaleScaledLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaHessianHesScaleScaled
        name = "adahessian_hesscale_scaled"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)