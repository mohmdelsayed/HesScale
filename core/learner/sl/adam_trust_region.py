from core.learner.sl.learner import Learner
from core.optim.adam_trust_region import AdamTrustRegionG

class AdamTrustRegionGLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdamTrustRegionG
        name = "adam_trust_region_g"
        super().__init__(name, network, optimizer, optim_kwargs)
