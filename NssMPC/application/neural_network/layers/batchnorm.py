#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch

from NssMPC.application.neural_network.functional.functional import torch2share
from NssMPC.config import data_type


class SecBatchNorm2d(torch.nn.Module):
    """
    A special batch normalization method.

    The implementation of this class is mainly based on the paper Sonic.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True,
                 track_running_stats: bool = True, device=None):
        """
        Initializes the SecBatchNorm2d layer.

        Args:
            num_features: The number of input features, usually equivalent to the number of channels.
            eps: A small value used to prevent division by zero. Defaults to 1e-5.
            momentum: The momentum used to update the running mean and variance. Defaults to 0.1.
            affine: Whether to use learnable affine transformations. Defaults to True.
            track_running_stats: Whether to track runtime statistics (mean and variance). Defaults to True.
            device (str, optional): Specify the device where the parameter is stored.

        Examples:
            >>> bn = SecBatchNorm2d(num_features=16)
        """
        super(SecBatchNorm2d, self).__init__()
        self.weight = torch.nn.Parameter(torch.zeros([num_features], dtype=data_type), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros([num_features], dtype=data_type), requires_grad=False)

    def forward(self, x):
        """
        Forward propagation function.

        Args:
            x (ArithmeticSecretSharing or ReplicatedSecretSharing): The data tensor to be processed.

        Returns:
            torch.Tensor: The processed data tensor.

        Examples:
            >>> output = bn(input_tensor)
        """
        epsilon1 = torch2share(self.weight, x.__class__, x.dtype).unsqueeze(1).unsqueeze(2)
        epsilon2 = torch2share(self.bias, x.__class__, x.dtype).unsqueeze(1).unsqueeze(2)
        return (x * epsilon1) + epsilon2
