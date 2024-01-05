import torch
from backpack import backpack, extend
from optimizers.adahesscale import AdaHesScale
from optimizers.adahesscalegn import AdaHesScaleGN
from hesscale.core.additional_losses import SoftmaxPPOLoss
from hesscale import HesScale, HesScaleGN

# Set seed for reproducibility
torch.manual_seed(42)

# Define hyperparameters
n_input = 4
n_output = 10
batch_size = 32
hidden_size = 32
lr = 0.00001
T = 1

# Define model
actor = torch.nn.Sequential(
    torch.nn.Linear(n_input, hidden_size),
    torch.nn.Tanh(),
    torch.nn.Linear(hidden_size, n_output),
)

# Define loss function
loss_fn = SoftmaxPPOLoss(reduction="mean", epsilon=0.2)

# optimizer = AdaHesScaleGN(
optimizer = AdaHesScale(
# optimizer = torch.optim.Adam(
    [
        {"params": actor.parameters(), "lr": lr},
    ],
)

extend(actor)
extend(loss_fn)

# Define input and target
state = torch.randn(batch_size, n_input)
advantage = torch.randn(batch_size, 1)
old_action_log_probs = torch.randn(batch_size, 1)
actions = torch.randint(0, n_output, (batch_size, 1))

for i in range(T):
    action_prefs = actor(state)
    loss = loss_fn(action_prefs, old_action_log_probs, advantage, actions)
    with backpack(HesScale(), HesScaleGN()):
        loss.backward()
    optimizer.step()

    for name, param in actor.named_parameters():
        if param.grad is not None:
            print(name, param.hesscale_gn.shape)