# -*- coding: utf-8 -*-
"""All torch modules that are used by the testproblems."""

import torch
from torch import nn
from torch.nn import functional as F
from deepobs.pytorch.testproblems.testproblems_utils import (
    _truncated_normal_init,
    tfmaxpool2d,
    tfconv2d,
)


def flatten():
    return torch.nn.Flatten()

def mean_allcnnc():
    """The all convolution layer implementation of torch.mean()."""
    # TODO implement pre forward hook to adapt to arbitary image size for other data sets than cifar100
    return nn.Sequential(nn.AvgPool2d(kernel_size=(2, 2)), flatten())


class net_mnist_logreg(nn.Sequential):
    def __init__(self, num_outputs):
        super(net_mnist_logreg, self).__init__()

        self.add_module("flatten", flatten())
        self.add_module("dense", nn.Linear(in_features=784, out_features=num_outputs))

        # init
        nn.init.constant_(self.dense.bias, 0.0)
        nn.init.constant_(self.dense.weight, 0.0)

class net_mnist_logreg_obd(nn.Sequential):
    def __init__(self, num_outputs):
        super(net_mnist_logreg_obd, self).__init__()

        self.add_module("flatten", flatten())
        self.add_module("dense", nn.Linear(in_features=784, out_features=num_outputs))
        self.add_module("logsoftmax", nn.LogSoftmax())

        # init
        nn.init.constant_(self.dense.bias, 0.0)
        nn.init.constant_(self.dense.weight, 0.0)

class net_cifar10_3c3d(nn.Sequential):
    """Basic conv net for cifar10/100. The network consists of
      - thre conv layers with ELUs, each followed by max-pooling
      - two fully-connected layers with ``512`` and ``256`` units and ELU activation
      - output layer with softmax
    The weight matrices are initialized using Xavier initialization and the biases
    are initialized to ``0.0``."""

    def __init__(self, num_outputs, use_tanh=True):
        """Args:
        num_outputs (int): The numer of outputs (i.e. target classes)."""
        super(net_cifar10_3c3d, self).__init__()
        self.name = 'net_cifar10_3c3d'
        activation = nn.Tanh if use_tanh else nn.ELU

        self.add_module(
            "conv1", tfconv2d(in_channels=3, out_channels=64, kernel_size=5)
        )
        self.add_module("act1", activation())
        self.add_module(
            "maxpool1", tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type = None)
        )

        self.add_module(
            "conv2", tfconv2d(in_channels=64, out_channels=96, kernel_size=3)
        )
        self.add_module("act2", activation())
        self.add_module(
            "maxpool2", tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type = None)
        )

        self.add_module(
            "conv3",
            tfconv2d(
                in_channels=96, out_channels=128, kernel_size=3, tf_padding_type = None
            ),
        )
        self.add_module("act3", activation())
        self.add_module(
            "maxpool3", tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type = None)
        )

        self.add_module("flatten", flatten())

        self.add_module("dense1", nn.Linear(in_features=1 * 1 * 128, out_features=512))
        self.add_module("act4", activation())
        self.add_module("dense2", nn.Linear(in_features=512, out_features=256))
        self.add_module("act5", activation())
        self.add_module("dense3", nn.Linear(in_features=256, out_features=num_outputs))

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_normal_(module.weight)

            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)

class net_cifar10_3c3d_obd(nn.Sequential):
    """Basic conv net for cifar10/100. The network consists of
      - thre conv layers with ELUs, each followed by max-pooling
      - two fully-connected layers with ``512`` and ``256`` units and ELU activation
      - output layer with softmax
    The weight matrices are initialized using Xavier initialization and the biases
    are initialized to ``0.0``."""

    def __init__(self, num_outputs, use_tanh=True):
        """Args:
        num_outputs (int): The numer of outputs (i.e. target classes)."""
        super(net_cifar10_3c3d_obd, self).__init__()
        activation = nn.Tanh if use_tanh else nn.ELU

        self.add_module(
            "conv1", tfconv2d(in_channels=3, out_channels=64, kernel_size=5)
        )
        self.add_module("act1", activation())
        self.add_module(
            "maxpool1", tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type = None)
        )

        self.add_module(
            "conv2", tfconv2d(in_channels=64, out_channels=96, kernel_size=3)
        )
        self.add_module("act2", activation())
        self.add_module(
            "maxpool2", tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type = None)
        )

        self.add_module(
            "conv3",
            tfconv2d(
                in_channels=96, out_channels=128, kernel_size=3, tf_padding_type = None
            ),
        )
        self.add_module("act3", activation())
        self.add_module(
            "maxpool3", tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type = None)
        )

        self.add_module("flatten", flatten())

        self.add_module("dense1", nn.Linear(in_features=1 * 1 * 128, out_features=512))
        self.add_module("act4", activation())
        self.add_module("dense2", nn.Linear(in_features=512, out_features=256))
        self.add_module("act5", activation())
        self.add_module("dense3", nn.Linear(in_features=256, out_features=num_outputs))
        self.add_module("logsoftmax", nn.LogSoftmax())

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_normal_(module.weight)

            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)


class net_mnist_2c2d(nn.Sequential):
    """  Basic conv net for (Fashion-)MNIST. The network has been adapted from the `TensorFlow tutorial\
  <https://www.tensorflow.org/tutorials/estimators/cnn>`_ and consists of

    - two conv layers with ELUs, each followed by max-pooling
    - one fully-connected layers with ELUs
    - output layer with softmax

  The weight matrices are initialized with truncated normal (standard deviation
  of ``0.05``) and the biases are initialized to ``0.05``."""

    def __init__(self, num_outputs, use_tanh=True):
        """Args:
        num_outputs (int): The numer of outputs (i.e. target classes)."""

        super(net_mnist_2c2d, self).__init__()
        activation = nn.Tanh if use_tanh else nn.ELU

        self.add_module(
            "conv1",
            tfconv2d(
                in_channels=1, out_channels=32, kernel_size=5, tf_padding_type = None
            ),
        )
        self.add_module("act1", activation())
        self.add_module(
            "max_pool1", tfmaxpool2d(kernel_size=2, stride=2, tf_padding_type = None)
        )

        self.add_module(
            "conv2",
            tfconv2d(
                in_channels=32, out_channels=64, kernel_size=5, tf_padding_type = None
            ),
        )
        self.add_module("act2", activation())
        self.add_module(
            "max_pool2", tfmaxpool2d(kernel_size=2, stride=2, tf_padding_type = None)
        )

        self.add_module("flatten", flatten())

        self.add_module("dense1", nn.Linear(in_features=4 * 4 * 64, out_features=1024))
        self.add_module("act3", activation())

        self.add_module("dense2", nn.Linear(in_features=1024, out_features=num_outputs))

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.05)
                module.weight.data = _truncated_normal_init(
                    module.weight.data, mean=0, stddev=0.05
                )

            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.05)
                module.weight.data = _truncated_normal_init(
                    module.weight.data, mean=0, stddev=0.05
                )


class net_mnist_2c2d_obd(nn.Sequential):
    """  Basic conv net for (Fashion-)MNIST. The network has been adapted from the `TensorFlow tutorial\
  <https://www.tensorflow.org/tutorials/estimators/cnn>`_ and consists of

    - two conv layers with ELUs, each followed by max-pooling
    - one fully-connected layers with ELUs
    - output layer with softmax

  The weight matrices are initialized with truncated normal (standard deviation
  of ``0.05``) and the biases are initialized to ``0.05``."""

    def __init__(self, num_outputs, use_tanh=True):
        """Args:
        num_outputs (int): The numer of outputs (i.e. target classes)."""

        super(net_mnist_2c2d_obd, self).__init__()
        activation = nn.Tanh if use_tanh else nn.ELU

        self.add_module(
            "conv1",
            tfconv2d(
                in_channels=1, out_channels=32, kernel_size=5, tf_padding_type = None
            ),
        )
        self.add_module("act1", activation())
        self.add_module(
            "max_pool1", tfmaxpool2d(kernel_size=2, stride=2, tf_padding_type = None)
        )

        self.add_module(
            "conv2",
            tfconv2d(
                in_channels=32, out_channels=64, kernel_size=5, tf_padding_type = None
            ),
        )
        self.add_module("act2", activation())
        self.add_module(
            "max_pool2", tfmaxpool2d(kernel_size=2, stride=2, tf_padding_type = None)
        )

        self.add_module("flatten", flatten())

        self.add_module("dense1", nn.Linear(in_features=4 * 4 * 64, out_features=1024))
        self.add_module("act3", activation())

        self.add_module("dense2", nn.Linear(in_features=1024, out_features=num_outputs))
        self.add_module("logsoftmax", nn.LogSoftmax())

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.05)
                module.weight.data = _truncated_normal_init(
                    module.weight.data, mean=0, stddev=0.05
                )

            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.05)
                module.weight.data = _truncated_normal_init(
                    module.weight.data, mean=0, stddev=0.05
                )

class net_cifar100_allcnnc(nn.Sequential):
    def __init__(self, num_outputs=100, use_tanh=True):
        super(net_cifar100_allcnnc, self).__init__()
        self.name = 'net_cifar100_allcnnc'

        activation = nn.Tanh if use_tanh else nn.ELU

        self.add_module("dropout1", nn.Dropout(p=0.2))

        self.add_module(
            "conv1",
            tfconv2d(
                in_channels=3, out_channels=96, kernel_size=3, tf_padding_type = None
            ),
        )
        self.add_module("act1", activation())
        self.add_module(
            "conv2",
            tfconv2d(
                in_channels=96, out_channels=96, kernel_size=3, tf_padding_type = None
            ),
        )
        self.add_module("act2", activation())
        self.add_module(
            "conv3",
            tfconv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=(2, 2),
                tf_padding_type = None,
            ),
        )
        self.add_module("act3", activation())

        self.add_module("dropout2", nn.Dropout(p=0.5))

        self.add_module(
            "conv4",
            tfconv2d(
                in_channels=96, out_channels=192, kernel_size=3, tf_padding_type = None
            ),
        )
        self.add_module("act4", activation())
        self.add_module(
            "conv5",
            tfconv2d(
                in_channels=192, out_channels=192, kernel_size=3, tf_padding_type = None
            ),
        )
        self.add_module("act5", activation())
        self.add_module(
            "conv6",
            tfconv2d(
                in_channels=192,
                out_channels=192,
                kernel_size=3,
                stride=(2, 2),
                tf_padding_type = None,
            ),
        )
        self.add_module("act6", activation())

        self.add_module("dropout3", nn.Dropout(p=0.5))

        self.add_module(
            "conv7", tfconv2d(in_channels=192, out_channels=192, kernel_size=3)
        )
        self.add_module("act7", activation())
        self.add_module(
            "conv8",
            tfconv2d(
                in_channels=192, out_channels=192, kernel_size=1, tf_padding_type = None
            ),
        )
        self.add_module("act8", activation())
        self.add_module(
            "conv9",
            tfconv2d(
                in_channels=192, out_channels=num_outputs, kernel_size=1, tf_padding_type = None
            ),
        )
        self.add_module("act9", activation())

        self.add_module("mean", mean_allcnnc())

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.1)
                nn.init.xavier_normal_(module.weight)


class net_imagenet_allcnnc(nn.Sequential):
    def __init__(self, use_tanh=True):
        super(net_imagenet_allcnnc, self).__init__()

        activation = nn.Tanh if use_tanh else nn.ReLU

        self.add_module("dropout1", nn.Dropout(p=0.2))

        self.add_module(
            "conv1",
            tfconv2d(
                in_channels=3, out_channels=96, kernel_size=3, tf_padding_type = None
            ),
        )
        self.add_module("act1", activation())
        self.add_module(
            "conv2",
            tfconv2d(
                in_channels=96, out_channels=96, kernel_size=3, tf_padding_type = None
            ),
        )
        self.add_module("act2", activation())
        self.add_module(
            "conv3",
            tfconv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=(2, 2),
                tf_padding_type = None,
            ),
        )
        self.add_module("act3", activation())

        self.add_module("dropout2", nn.Dropout(p=0.5))

        self.add_module(
            "conv4",
            tfconv2d(
                in_channels=96, out_channels=192, kernel_size=3, tf_padding_type = None
            ),
        )
        self.add_module("act4", activation())
        self.add_module(
            "conv5",
            tfconv2d(
                in_channels=192, out_channels=192, kernel_size=3, tf_padding_type = None
            ),
        )
        self.add_module("act5", activation())
        self.add_module(
            "conv6",
            tfconv2d(
                in_channels=192,
                out_channels=192,
                kernel_size=3,
                stride=(2, 2),
                tf_padding_type = None,
            ),
        )
        self.add_module("act6", activation())

        self.add_module("dropout3", nn.Dropout(p=0.5))

        self.add_module(
            "conv7", tfconv2d(in_channels=192, out_channels=192, kernel_size=3)
        )
        self.add_module("act7", activation())
        self.add_module(
            "conv8",
            tfconv2d(
                in_channels=192, out_channels=192, kernel_size=1, tf_padding_type = None
            ),
        )
        self.add_module("act8", activation())
        self.add_module(
            "conv9",
            tfconv2d(
                in_channels=192, out_channels=1000, kernel_size=1, tf_padding_type = None
            ),
        )
        self.add_module("act9", activation())

        self.add_module("mean", mean_allcnnc())

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.1)
                nn.init.xavier_normal_(module.weight)


class net_mlp(nn.Sequential):
    """A basic MLP architecture. The network is build as follows:

    - Four fully-connected layers with ``1000``, ``500``,``100`` and ``num_outputs``
      units per layer, where ``num_outputs`` is the number of ouputs (i.e. class labels).
    - The first three layers use Tanh activation, and the last one a softmax
      activation.
    - The biases are initialized to ``0.0`` and the weight matrices with
      truncated normal (standard deviation of ``3e-2``)"""

    def __init__(self, num_outputs, use_tanh=True):
        super(net_mlp, self).__init__()
        activation = nn.Tanh if use_tanh else nn.ELU
        self.add_module("flatten", flatten())
        self.add_module("dense1", nn.Linear(784, 1000))
        self.add_module("act1", activation())
        self.add_module("dense2", nn.Linear(1000, 500))
        self.add_module("act2", activation())
        self.add_module("dense3", nn.Linear(500, 100))
        self.add_module("act3", activation())
        self.add_module("dense4", nn.Linear(100, num_outputs))

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                module.weight.data = _truncated_normal_init(
                    module.weight.data, mean=0, stddev=3e-2
                )

class net_mlp_obd(nn.Sequential):
    """A basic MLP architecture for Negative Log Likelihood loss. The network is build as follows:

    - Four fully-connected layers with ``1000``, ``500``,``100`` and ``num_outputs``
      units per layer, where ``num_outputs`` is the number of ouputs (i.e. class labels).
    - The first three layers use Tanh activation, and the last one a softmax
      activation.
    - The biases are initialized to ``0.0`` and the weight matrices with
      truncated normal (standard deviation of ``3e-2``)"""

    def __init__(self, num_outputs, use_tanh=True):
        super(net_mlp_obd, self).__init__()
        activation = nn.Tanh if use_tanh else nn.ELU
        self.add_module("flatten", flatten())
        self.add_module("dense1", nn.Linear(784, 1000))
        self.add_module("act1", activation())
        self.add_module("dense2", nn.Linear(1000, 500))
        self.add_module("act2", activation())
        self.add_module("dense3", nn.Linear(500, 100))
        self.add_module("act3", activation())
        self.add_module("dense4", nn.Linear(100, num_outputs))
        self.add_module("logsoftmax", nn.LogSoftmax())

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                module.weight.data = _truncated_normal_init(
                    module.weight.data, mean=0, stddev=3e-2
                )
