from core.learner.sl.learner import Learner
from core.optim.adam_overshooting import AdamWithOvershootingPrevention

class AdamWithOvershootingPreventionLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdamWithOvershootingPrevention
        name = "adam_with_overshooting_prevention"
        super().__init__(name, network, optimizer, optim_kwargs)