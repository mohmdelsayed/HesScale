from torch.nn.modules.loss import _Loss
import torch
from torch import Tensor
from torch.nn import functional as F

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
