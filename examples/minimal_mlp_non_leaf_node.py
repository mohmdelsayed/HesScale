import torch
from hesscale import HesScale
from backpack import backpack, extend


def example():
    hidden_units = 128
    n_obs = 6
    n_classes = 10
    lr = 0.0004
    T = 1
    batch_size = 1

    model = torch.nn.Sequential(
        torch.nn.Linear(n_obs, hidden_units),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_units, n_classes),
    )
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    extend(model)
    extend(loss_func)

    for i in range(T):
        inputs = torch.randn((batch_size, n_obs))
        target_class = torch.randint(0, n_classes, (batch_size,))

        intermediate_variable = model[:2](inputs)
        intermediate_variable.retain_grad()

        prediction = model[-1](intermediate_variable)

        loss = loss_func(prediction, target_class)

        optimizer.zero_grad()
        with backpack(HesScale()):
            loss.backward()

        optimizer.step()

        print("Non-leaf node size: ", intermediate_variable.hesscale_.shape)
        for (name, param) in model.named_parameters():
            print(name, param.hesscale.shape)


if __name__ == "__main__":
    example()
