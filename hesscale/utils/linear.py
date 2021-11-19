from torch import einsum
from hesscale.derivatives import LOSS

def extract_weight_diagonal(module, backproped, sum_batch=True):
    if sum_batch:
        equation = "vno,ni->oi"
    else:
        equation = "vno,ni->noi"

    if LOSS in backproped:
        return einsum(equation, (backproped[0], module.input0 ** 2))
    else:
        return einsum(equation, (backproped[0] + backproped[1], module.input0 ** 2))


def extract_bias_diagonal(module, backproped, sum_batch=True):
    if sum_batch:
        equation = "vno->o"
    else:
        equation = "vno->no"
    if LOSS in backproped:
        return einsum(equation, (backproped[0]))
    else:
        return einsum(equation, (backproped[0] + backproped[1]))
