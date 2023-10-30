import unittest
import torch
from core.optim.sgd import SGD

class TestSGD(unittest.TestCase):
    
    def test_sgd_optimizer(self):
        # Test case: Test SGD optimizer with a simple linear regression model
        model = torch.nn.Linear(1, 1)
        # initialize weights:
        model.weight.data = torch.tensor([[0.0]])
        model.bias.data = torch.tensor([0.0])

        optimizer = SGD(model.parameters(), lr=0.1)
        loss_fn = torch.nn.MSELoss()
        
        # Train the model for a few epochs
        x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
        for _ in range(500):
            optimizer.zero_grad()
            def closure():
                y_pred = model(x_train)
                return loss_fn(y_pred, y_train), {}
            optimizer.step(closure=closure)
        
        # Check if the model has learned the correct weights
        self.assertAlmostEqual(model.weight.item(), 2.0, delta=0.02)
        self.assertAlmostEqual(model.bias.item(), 0.0, delta=0.02)
        
if __name__ == '__main__':
    unittest.main()