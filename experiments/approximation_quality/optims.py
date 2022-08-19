from torch.optim import SGD, Adam

from experiments.approximation_quality.optimizers.ada_hessian import Adahessian
from experiments.approximation_quality.optimizers.exact_diag_hess import ExactHessDiagOptimizer
from experiments.approximation_quality.optimizers.ggn import GGNExactOptimizer
from experiments.approximation_quality.optimizers.ggn_mc import GGNMCOptimizer
from experiments.approximation_quality.optimizers.hesscale import HesScaleOptimizer

optimizers = {
    "hesscale": HesScaleOptimizer,
    "ggn": GGNExactOptimizer,
    "ggn_mc": GGNMCOptimizer,
    "adahess": Adahessian,
    "exact_h": ExactHessDiagOptimizer,
    "adam": Adam,
    "sgd": SGD,
}
