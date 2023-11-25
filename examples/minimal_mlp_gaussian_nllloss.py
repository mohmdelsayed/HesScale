import torch
from backpack import backpack, extend
from optimizers.adahesscale import AdaHesScale
from optimizers.adahesscalegn import AdaHesScaleGN
from hesscale.core.additional_losses import GaussianNLLLossMu, GaussianNLLLossVar
from hesscale.core.additional_activations import Exponential
from hesscale import HesScale, HesScaleGN

# Set seed for reproducibility
torch.manual_seed(42)

# Define hyperparameters
n_input = 2
n_output = 1
batch_size = 32
hidden_size = 32
lr = 0.00001
T = 10000

# Define model
mu_net = torch.nn.Sequential(
    torch.nn.Linear(n_input, hidden_size),
    torch.nn.Tanh(),
    torch.nn.Linear(hidden_size, n_output),
)

var_net = torch.nn.Sequential(
    torch.nn.Linear(n_input, hidden_size),
    torch.nn.Tanh(),
    torch.nn.Linear(hidden_size, n_output),
    Exponential(),
)

# Define loss function
loss_fn_mu = GaussianNLLLossMu(reduction="mean", eps=1e-6)
loss_fn_var = GaussianNLLLossVar(reduction="mean", eps=1e-6)

# optimizer = AdaHesScale(
optimizer = AdaHesScaleGN(
# optimizer = torch.optim.Adam(
    [
        {"params": var_net.parameters(), "lr": lr},
        {"params": mu_net.parameters(), "lr": lr},
    ],
)

extend(mu_net)
extend(var_net)
extend(loss_fn_mu)
extend(loss_fn_var)

# Define input and target
input = torch.randn(batch_size, n_input)
target = torch.randn(batch_size, n_output) * 0.0 + 0.5

for i in range(T):
    # Forward pass 
    predicted_mean = mu_net(input)
    predicted_var = var_net(input)

    # Compute loss
    loss_mean = loss_fn_mu(predicted_mean, predicted_var, target)
    loss_var = loss_fn_var(predicted_var, predicted_mean, target)

    with backpack(HesScaleGN(), HesScale()):
        loss_mean.backward()
        loss_var.backward()
    optimizer.step()
    
    # expected: mu=0.5, var=0.0
    print(f"predicted_mean: {round(predicted_mean.detach().mean(0).item(), 3)}", f"predicted_var: {round(predicted_var.detach().squeeze(0).mean(0).item(), 4)}")
