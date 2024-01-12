import torch
from backpack import backpack, extend
from optimizers.adahesscale import AdaHesScale
from optimizers.adahesscalegn import AdaHesScaleGN
from hesscale.core.additional_losses import GaussianNLLLossMuPPO, GaussianNLLLossVarPPO
from hesscale.core.additional_activations import Exponential
from hesscale import HesScale, HesScaleGN

# Set seed for reproducibility
torch.manual_seed(42)

# Define hyperparameters
n_input = 2
n_output = 6
batch_size = 32
hidden_size = 32
lr = 0.0001
T = 1

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
loss_fn_mu = GaussianNLLLossMuPPO(reduction="mean", eps=1e-6)
loss_fn_var = GaussianNLLLossVarPPO(reduction="mean", eps=1e-6)

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
state = torch.randn(batch_size, n_input)
advantages = torch.randn(batch_size, 1)
old_action_probs = torch.randn(batch_size, 1)
actions = torch.randint(0, n_output, (batch_size, 1))

for i in range(T):
    # Forward pass 
    predicted_means = mu_net(state)
    predicted_vars = var_net(state)

    # Compute loss
    loss_mean = loss_fn_mu(predicted_means, predicted_vars, actions, old_action_probs, advantages)
    loss_var = loss_fn_var(predicted_vars, predicted_means, actions, old_action_probs, advantages)
    optimizer.zero_grad()

    with backpack(HesScaleGN(), HesScale()):
        loss_mean.backward()
        loss_var.backward()
    optimizer.step()

    for name, param in mu_net.named_parameters():
        if param.grad is not None:
            print(name, param.hesscale_gn.shape)