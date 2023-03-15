import warnings
from backpack.extensions.mat_to_mat_jac_base import MatToJacMat

class BaseModuleHesScale(MatToJacMat):
    def __init__(self, derivatives, params=None):
        super().__init__(derivatives=derivatives, params=params)

    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):
        if isinstance(backproped, list):
            M_list = [
                self.derivatives.diag_hessian(module, grad_inp, grad_out, M)
                for M in backproped
            ]
            return list(M_list)
        else:
            return self.derivatives.diag_hessian(module, grad_inp, grad_out, backproped)
