import torch.nn as nn

class FCNReLU(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=300):
        super(FCNReLU, self).__init__()
        self.name = "fcn_relu"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", nn.ReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units // 2))
        self.add_module("act_2", nn.ReLU())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units // 2, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name
    

class FCNReLUSmall(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=32):
        super(FCNReLUSmall, self).__init__()
        self.name = "fcn_relu_small"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units, bias=False))
        self.add_module("act_1", nn.ReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units, bias=False))
        self.add_module("act_2", nn.ReLU())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units, out_features=n_outputs, bias=False))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name


class FCNReLUSmallSoftmax(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=32):
        super(FCNReLUSmallSoftmax, self).__init__()
        self.name = "fcn_relu_small"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units, bias=False))
        self.add_module("act_1", nn.ReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units, bias=False))
        self.add_module("act_2", nn.ReLU())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units, out_features=n_outputs, bias=False))
        self.add_module("log_softmax", nn.LogSoftmax(dim=1))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name
    
if __name__ == "__main__":
    net = FCNReLU()
    print(net)
