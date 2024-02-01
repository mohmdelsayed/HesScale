from core.learner.sl.learner import Learner
from core.optim.adam_with_hesscale_scaling import AdamHesScale

class AdamScaledHesScaleLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdamHesScale
        name = "adam_scaled_hesscale"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)