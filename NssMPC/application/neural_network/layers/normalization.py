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
    * The implementation of this class is mainly based on the `paper Sigma <https://eprint.iacr.org/2023/1269.pdf>`_.
    """

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        """
        Initialize the SecLayerNorm class.

        :param normalized_shape: The dimension that needs normalization is usually the last dimension of the input tensor.
        :type normalized_shape: int
        :param eps: Constants are used to prevent division by zero errors (default is **1e-05**).
        :type eps: float
        :param elementwise_affine: The parameter indicates whether to use learnable scaling and translation (weight and bias) (the default value set to **True**).
        :type elementwise_affine: bool

        ATTRIBUTES:
            * **normalized_shape** (*int*): The dimension that needs normalization is usually the last dimension of the input tensor.
            * **eps** (*float*): Constants are used to prevent division by zero errors (default is **1e-05**).
            * **weight** (*torch.Tensor*): The weights of neural networks.
            * **bias** (*torch.Tensor*): bias term
            * **scale** (*torch.Tensor*): the scale of the tensor.
            * **zero_point** (*int*): The weights of neural networks.
            * **elementwise_affine** (*torch.Tensor*): The weights of neural networks.

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
        The forward propagation process.

        Start by summing the input ``x`` along the last dimension, divided by the mean obtained by ``self.normalized_shape``.
        The input data is then centralized, i.e. the mean is subtracted from each element. After calculating the
        reciprocal of the square root of variance ``inv_sqrt_variance``, we use the :func:`~NssMPC.application.neural_network.functional.functional.torch2share` function to obtain weight
        and bias. Finally, we multiply the centralized data by the reciprocal of the standard deviation and apply
        scaling (if ``elementwise_affine`` is enabled). And then you add the ``bias``.

        :param x: Input tensor, typically coming from the output of the previous layer.
        :type x: ArithmeticSecretSharing or ReplicatedSecretSharing
        :return: A tensor after layer normalization
        :rtype: ArithmeticSecretSharing or ReplicatedSecretSharing
        """
        mean = x.sum(dim=-1).unsqueeze(-1) / self.normalized_shape

        z = x - mean
        inv_sqrt_variance = x.__class__.rsqrt(((z * z).sum(dim=-1).unsqueeze(-1)) / self.normalized_shape)

        weight = torch2share(self.weight, x.__class__, x.dtype)
        bias = torch2share(self.bias, x.__class__, x.dtype)

        z = z * inv_sqrt_variance * weight + bias

        return z
