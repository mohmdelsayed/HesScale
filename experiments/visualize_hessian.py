import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import warnings
warnings.filterwarnings("ignore")
import numpy as np

def hessian(grads, params):
    # Calculate Hessian using Hessian-vector product with one-hot vectors
    vs = []
    for p in params:
        v = torch.zeros_like(p)
        vs.append(v)
    vectorized_v = parameters_to_vector(vs)
    hvps = []
    for i in range(vectorized_v.shape[0]):
        vectorized_v[i] = 1
        vector_to_parameters(vectorized_v, vs)
        hvp = torch.autograd.grad(grads, params, vs, retain_graph=True)
        vectorized_v = parameters_to_vector(vs)
        vectorized_v[i] = 0
        hvp = torch.cat([e.flatten() for e in hvp])
        hvps.append(hvp)
    hessian_matrix = torch.stack(hvps)
    return hessian_matrix

def compute_diag_dominance(hessian_matrix):
    return hessian_matrix.diag().norm() / hessian_matrix.norm()

def train_model(model, mnist_train, n_iterations=1000, batch_size=16):
    train_loader = iter(
        torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    )

    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)
    for _ in range(n_iterations):
        try:
            x, y = next(train_loader)
        except:
            train_loader = iter(
                torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
            )
            x, y = next(train_loader)
        y_hat = model(x)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":

    n_seeds = 100

    mnist_train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(lambda x: torch.flatten(x)),
            ]
        ),
    )

    # train loader mnist:
    train_loader = iter(
        torch.utils.data.DataLoader(mnist_train, batch_size=1000, shuffle=True)
    )

    n_input = 784
    n_output = 10
    Hs_before = []
    Hs_after = []

    for seed in range(n_seeds):

        torch.manual_seed(seed)

        model = torch.nn.Sequential(
            torch.nn.Linear(n_input, 32, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(32, n_output, bias=False),
        )

        loss_fn = torch.nn.CrossEntropyLoss()

        # get data
        x, y = next(train_loader)
        y_hat = model(x)
        loss_before = loss_fn(y_hat, y)

        # compute Hessian before training:
        grads = torch.autograd.grad(
            loss_before, list(model.parameters())[1:], create_graph=True
        )
        hessian_matrix_before = hessian(grads, list(model.parameters())[1:])
        Hs_before.append(hessian_matrix_before)

        train_model(model, mnist_train)

        # compute Hessian after training:
        y_hat = model(x)
        loss_after = loss_fn(y_hat, y)
        grads = torch.autograd.grad(
            loss_after, list(model.parameters())[1:], create_graph=True
        )
        hessian_matrix_after = hessian(grads, list(model.parameters())[1:])
        Hs_after.append(hessian_matrix_after)

        print(
            "Seed:",
            seed,
            "Loss Before: ",
            loss_before.item(),
            "Loss After: ",
            loss_after.item(),
        )

    Hs_before_avg = torch.stack(Hs_before).abs().mean(dim=0)
    Hs_after_avg = torch.stack(Hs_after).abs().mean(dim=0)

    cmap = plt.cm.get_cmap("RdBu")
    plt.imshow(
        Hs_before_avg.detach().numpy(),
        vmin=Hs_before_avg.min().item(),
        vmax=Hs_before_avg.max().item(),
        cmap=cmap,
    )
    plt.colorbar()
    plt.title("Magitudes of Hessian matrix before training")
    plt.savefig("H_before.pdf", bbox_inches="tight")
    print("Diag Dominance Before: ", compute_diag_dominance(Hs_before_avg))

    plt.clf()

    plt.imshow(
        Hs_after_avg.detach().numpy(),
        vmin=Hs_after_avg.min().item(),
        vmax=Hs_after_avg.max().item(),
        cmap=cmap,
    )
    plt.title("Magitudes of Hessian matrix after training")
    plt.colorbar()
    plt.savefig("H_after.pdf", bbox_inches="tight")
    print("Diag Dominance After: ", compute_diag_dominance(Hs_after_avg))
