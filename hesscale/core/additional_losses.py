from torch.nn.modules.loss import _Loss
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.distributions import Normal

class SoftmaxNLLLoss(_Loss):
      def __init__(self, reduction = 'mean'):
          self.reduction = reduction
          self.ignore_index = -100
          self.weight = None
          super(SoftmaxNLLLoss, self).__init__(reduction=reduction)

      def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """
            Arguments:
                input (Tensor): A 2D tensor of shape (batch_size, num_classes/num_actions). 
                                Represents the prediction where each row corresponds to a specific batch and each column corresponds to a class or an action.

                target (Tensor): A 2D tensor of shape (batch_size, num_classes/num_actions). 
                                Represents the target where each row corresponds to a specific batch and each column corresponds to a class or an action.

            Returns:
                output (Tensor): A tensor of shape ().
        """
        log_softmax = F.log_softmax(prediction, dim=1)
        if self.reduction == 'mean':
            return -torch.mean(torch.einsum("ij,ij->i", log_softmax, target))
        elif self.reduction == 'sum':
            return -torch.sum(torch.einsum("ij,ij->i", log_softmax, target))
        else:
            raise ValueError("Invalid reduction type")

class SoftmaxPPOLoss(_Loss):
    def __init__(self, reduction = 'mean', epsilon = 0.2):
          self.reduction = reduction
          self.ignore_index = -100
          self.weight = None
          self.epsilon = epsilon
          super(SoftmaxPPOLoss, self).__init__(reduction=reduction)

    def forward(self, action_prefs: Tensor, old_action_prob: Tensor, advantage: Tensor, action: Tensor) -> Tensor:
        
        new_action_prob = self.get_prob(action_prefs, action)
        ratio = new_action_prob / old_action_prob

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage

        if self.reduction == 'mean':
            return -torch.min(surr1, surr2).mean()
        elif self.reduction == 'sum':
            return -torch.min(surr1, surr2).sum()
        else:
            raise ValueError("Invalid reduction type")
    
    def get_prob(self, action_prefs: Tensor, action:Tensor) -> Tensor:
        new_prob = F.softmax(action_prefs, dim=1)
        return torch.gather(new_prob, 1, action)
    
class GaussianNLLLossMu(_Loss):
    def __init__(self, full: bool = False, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super(GaussianNLLLossMu, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps

    def forward(self, input: Tensor, var: Tensor, target: Tensor, scaling: Tensor) -> Tensor:
        var = var.clone().detach()
        value = F.gaussian_nll_loss(input, target, var, full=self.full, eps=self.eps, reduction="none") * scaling
        if self.reduction == 'mean':
            return torch.mean(value)
        elif self.reduction == 'sum':
            return torch.sum(value)
        else:
            raise ValueError("Invalid reduction type")
    
class GaussianNLLLossVar(_Loss):
    def __init__(self, full: bool = False, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super(GaussianNLLLossVar, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps

    def forward(self, var: Tensor, input: Tensor, target: Tensor, scaling: Tensor) -> Tensor:
        input = input.clone().detach()
        value = F.gaussian_nll_loss(input, target, var, full=self.full, eps=self.eps, reduction="none") * scaling
        if self.reduction == 'mean':
            return torch.mean(value)
        elif self.reduction == 'sum':
            return torch.sum(value)
        else:
            raise ValueError("Invalid reduction type")

class GaussianNLLLossMuPPO(_Loss):
    def __init__(self, full: bool = False, eps: float = 1e-6, reduction: str = 'mean', epsilon=0.2) -> None:
        super(GaussianNLLLossMuPPO, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps
        self.epsilon = epsilon

    def forward(self, predicted_means: Tensor, predicted_vars: Tensor, actions: Tensor, old_probs: Tensor, advantage: Tensor) -> Tensor:

        probs = self.get_prob(predicted_means, predicted_vars.detach(), actions)

        ratio = probs / old_probs

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage

        if self.reduction == 'mean':
            return -torch.min(surr1, surr2).mean()
        elif self.reduction == 'sum':
            return -torch.min(surr1, surr2).sum()
        else:
            raise ValueError("Invalid reduction type")
    
    def get_prob(self, predicted_means: Tensor, predicted_vars: Tensor, actions:Tensor) -> Tensor:
        dist = Normal(predicted_means, predicted_vars)
        return dist.log_prob(actions).sum(1).exp().unsqueeze(1)
    
class GaussianNLLLossVarPPO(_Loss):
    def __init__(self, full: bool = False, eps: float = 1e-6, reduction: str = 'mean', epsilon=0.2) -> None:
        super(GaussianNLLLossVarPPO, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps
        self.epsilon = epsilon

    def forward(self, predicted_vars: Tensor, predicted_means: Tensor, actions: Tensor, old_probs: Tensor, advantage: Tensor) -> Tensor:

        probs = self.get_prob(predicted_means.detach(), predicted_vars, actions)
        ratio = probs / old_probs

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage

        if self.reduction == 'mean':
            return -torch.min(surr1, surr2).mean()
        elif self.reduction == 'sum':
            return -torch.min(surr1, surr2).sum()
        else:
            raise ValueError("Invalid reduction type")
    
    def get_prob(self, predicted_means: Tensor, predicted_vars: Tensor, actions:Tensor) -> Tensor:
        dist = Normal(predicted_means, predicted_vars)
        return dist.log_prob(actions).sum(1).exp().unsqueeze(1)
