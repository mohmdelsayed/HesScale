from torch import einsum
from hesscale.derivatives import LOSS, LINEAR, CONV, ACTIVATION


def extract_weight_diagonal(module, backproped, sum_batch=True):
    equation = "vno,ni->oi" if sum_batch else "vno,ni->noi"

    if LOSS in backproped or LINEAR in backproped or CONV in backproped:
        d2Ld2a = backproped[0]
    elif ACTIVATION in backproped:
        d2Ld2a = backproped[0] + backproped[1]
    else:
        raise NotImplementedError("No valid layer")

    return einsum(equation, (d2Ld2a, module.input0 ** 2))


def extract_bias_diagonal(module, backproped, sum_batch=True):
    equation = "vno->o" if sum_batch else "vno->no"

    if LOSS in backproped or LINEAR in backproped or CONV in backproped:
        d2Ld2a = backproped[0]
    elif ACTIVATION in backproped:
        d2Ld2a = backproped[0] + backproped[1]
    else:
        raise NotImplementedError("No valid layer")

    return einsum(equation, (d2Ld2a))
