from core.learner.sl.learner import Learner
from core.optim.sgd_with_scaling import SGDScaled, SGDScaledSqrt

class SGDScaledLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SGDScaled
        name = "sgd_scaled"
        super().__init__(name, network, optimizer, optim_kwargs)


class SGDScaledSqrtLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SGDScaledSqrt
        name = "sgd_scaled_sqrt"
        super().__init__(name, network, optimizer, optim_kwargs)