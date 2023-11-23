from torch.nn.modules.loss import _Loss
import torch
from torch import Tensor
from torch.nn import functional as F

class MultiLabelNLLoss(_Loss):
      def __init__(self, reduction = 'mean'):
          self.reduction = reduction
          super(MultiLabelNLLoss, self).__init__(reduction=reduction)

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
        if self.reduction == 'mean':
            return -torch.mean(torch.einsum("ij,ij->i", prediction, target))
        elif self.reduction == 'sum':
            return -torch.sum(torch.einsum("ij,ij->i", prediction, target))
        else:
            raise ValueError("Invalid reduction type")
        

class MultiLabelCrossEntropy(_Loss):
      def __init__(self, reduction = 'mean'):
          self.reduction = reduction
          super(MultiLabelCrossEntropy, self).__init__(reduction=reduction)

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
        

if __name__ == "__main__":

    import torch
    from torch import nn

    # network that outputs 10 classes with logsoftmax
    n_classes = 5
    n_inputs = 52
    batch_size = 100

    network = nn.Sequential(
        nn.Linear(n_inputs, n_classes),
        nn.LogSoftmax(dim=1)
    )

    loss_multi_label = MultiLabelNLLoss(reduction='mean')
    loss = torch.nn.NLLLoss(reduction='mean')

    input = torch.randn(batch_size, n_inputs)
    target = torch.randint(0, n_classes, (batch_size,))
    one_hot = F.one_hot(target)
    loss1 = loss_multi_label(network(input), one_hot.float())
    loss2 = loss(network(input), target)

    # check if the loss is the same
    assert torch.allclose(loss1, loss2)

    network = nn.Sequential(
        nn.Linear(n_inputs, n_classes),
    )

    loss_multi_label = MultiLabelCrossEntropy(reduction='mean')
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    input = torch.randn(batch_size, n_inputs)
    target = torch.randint(0, n_classes, (batch_size,))
    one_hot = F.one_hot(target)
    loss1 = loss_multi_label(network(input), one_hot.float())
    loss2 = loss(network(input), target)

    # check if the loss is the same
    assert torch.allclose(loss1, loss2)