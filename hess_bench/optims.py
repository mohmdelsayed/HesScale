from torch.optim import SGD, Adam

from hess_bench.optimizers.ada_hessian import Adahessian
from hess_bench.optimizers.exact_diag_hess import ExactHessDiagOptimizer
from hess_bench.optimizers.ggn import GGNExactOptimizer
from hess_bench.optimizers.ggn_mc import GGNMCOptimizer
from hess_bench.optimizers.hesscale import HesScaleOptimizer

optimizers = {
    "hesscale": HesScaleOptimizer,
    "ggn": GGNExactOptimizer,
    "ggn_mc": GGNMCOptimizer,
    "adahess": Adahessian,
    "exact_h": ExactHessDiagOptimizer,
    "adam": Adam,
    "sgd": SGD,
}
