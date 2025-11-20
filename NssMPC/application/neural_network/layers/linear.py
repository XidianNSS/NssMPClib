#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch
from NssMPC.application.neural_network.functional.functional import torch2share
from NssMPC.config import DEVICE


class SecLinear(torch.nn.Module):
    """
    The class is used to perform linear transformation operations.

    The implementation of this class is mainly based on the paper Sonic.
    """

    def __init__(self, input, output, weight=None, bias=None, device=DEVICE):
        """
        Initializes the SecLinear layer.

        Args:
            input (int): Dimension of the input feature.
            output (int): Dimension of the output feature.
            weight (torch.Tensor, optional): Optional weight parameters. Defaults to None.
            bias (torch.Tensor, optional): Optional bias parameters. Defaults to None.
            device (str, optional): The device where tensors are stored.

        Examples:
            >>> linear = SecLinear(input=10, output=5)
        """
        super(SecLinear, self).__init__()
        self.device = device
        self.weight = torch.nn.Parameter(torch.zeros([output, input], dtype=torch.int64), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros([output], dtype=torch.int64), requires_grad=False)

    def forward(self, x):
        """
        The forward propagation process.

        Args:
            x (ArithmeticSecretSharing): Input tensor.

        Returns:
            ArithmeticSecretSharing: The returned result calculated by a linear transformation.

        Examples:
            >>> output = linear(input_tensor)
        """
        weight = torch2share(self.weight, x.__class__, x.dtype).T
        bias = torch2share(self.bias, x.__class__, x.dtype)
        z = (x @ weight) + bias
        return z
