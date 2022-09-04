from turtle import back
from torch import einsum


def extract_weight_diagonal(module, backproped, sum_batch=True):
    equation = "vno,ni->oi" if sum_batch else "vno,ni->noi"
    return einsum(equation, (backproped, module.input0 ** 2))


def extract_bias_diagonal(module, backproped, sum_batch=True):
    equation = "vno->o" if sum_batch else "vno->no"
    return einsum(equation, (backproped))
