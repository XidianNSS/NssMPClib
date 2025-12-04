"""
The SecLayerNorm class is used to implement layer normalization.
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch

from NssMPC.application.neural_network.functional.functional import torch2share
from NssMPC.config import data_type


class SecLayerNorm(torch.nn.Module):
    """
    The SecLayerNorm class is used to implement layer normalization.

    The implementation of this class is mainly based on the paper Sigma.
    """

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        """
        Initializes the SecLayerNorm class.

        Args:
            normalized_shape (int or list): The dimension that needs normalization is usually the last dimension of the input tensor.
            eps (float, optional): Constants are used to prevent division by zero errors. Defaults to 1e-05.
            elementwise_affine (bool, optional): Indicates whether to use learnable scaling and translation (weight and bias). Defaults to True.

        Examples:
            >>> norm = SecLayerNorm(normalized_shape=10)
        """
        super(SecLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones([normalized_shape], dtype=data_type), requires_grad=False)
        self.bias = torch.nn.Parameter(
            torch.zeros([normalized_shape], dtype=data_type), requires_grad=False)
        self.scale = None
        self.zero_point = None
        self.elementwise_affine = elementwise_affine

    def forward(self, x):
        """
        The forward propagation process for layer normalization.

        Args:
            x (ArithmeticSecretSharing or ReplicatedSecretSharing): Input tensor, typically coming from the output of the previous layer.

        Returns:
            ArithmeticSecretSharing or ReplicatedSecretSharing: A tensor after layer normalization.

        Examples:
            >>> output = norm(input_tensor)
        """
        mean = x.sum(dim=-1).unsqueeze(-1) / self.normalized_shape

        z = x - mean
        inv_sqrt_variance = x.__class__.rsqrt(((z * z).sum(dim=-1).unsqueeze(-1)) / self.normalized_shape)

        weight = torch2share(self.weight, x.__class__, x.dtype)
        bias = torch2share(self.bias, x.__class__, x.dtype)

        z = z * inv_sqrt_variance * weight + bias

        return z
