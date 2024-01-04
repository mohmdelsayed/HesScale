from core.learner.sl.learner import Learner
from core.optim.adam_with_scaling import AdamScaled

class AdamScaledLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdamScaled
        name = "adam_scaled"
        super().__init__(name, network, optimizer, optim_kwargs)
