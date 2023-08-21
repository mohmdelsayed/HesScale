from core.learner.learner import Learner
from core.optim.adaggnmc import AdaGGNMC

class AdaGGNMCLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaGGNMC
        name = "adaggnmc"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)
