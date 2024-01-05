from core.learner.sl.learner import Learner
from core.optim.adam_with_scaling import AdamScaled, AdamScaledSqrt

class AdamScaledLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdamScaled
        name = "adam_scaled"
        super().__init__(name, network, optimizer, optim_kwargs)

class AdamScaledSqrtLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdamScaledSqrt
        name = "adam_scaled_sqrt"
        super().__init__(name, network, optimizer, optim_kwargs)
