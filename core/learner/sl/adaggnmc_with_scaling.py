from core.learner.sl.learner import Learner
from core.optim.adaggnmc_with_scaling import AdaGGNMCScaled

class AdaGGNMCScaledLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaGGNMCScaled
        name = "adaggnmc_scaled"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)