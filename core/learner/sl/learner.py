from backpack import extend
import torch
class Learner:
    def __init__(self, name, network, optimizer, optim_kwargs, extend=False):
        self.network_cls = network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optim_kwargs = optim_kwargs
        for k, v in optim_kwargs.items():
            if isinstance(v, str):
                optim_kwargs[k] = float(v)
        self.optimizer_cls = optimizer
        self.name = name
        self.extend = extend

    def __str__(self) -> str:
        return self.name

    def predict(self, input):
        output = self.network(input)
        return output

    def setup_task(self, task):
        if self.extend:
            self.network = extend(self.network_cls(num_outputs=task.n_outputs, use_tanh=False).to(self.device))
        else:
            self.network = self.network_cls(num_outputs=task.n_outputs, use_tanh=False).to(self.device)
        self.parameters = list(self.network.parameters())
        self.named_parameters = list(self.network.named_parameters())
        self.setup_optimizer()

    def setup_optimizer(self):
        self.optimizer = self.optimizer_cls(self.parameters, **self.optim_kwargs)

    def update_params(self, closure):
        return self.optimizer.step(closure)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def train(self, mode=True):
        self.network.train(mode)
