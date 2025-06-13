#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch
from NssMPC.application.neural_network.functional.functional import torch2share
from NssMPC.config import data_type


class SecBatchNorm2d(torch.nn.Module):
    """
    * The implementation of this class is mainly based on the `paper Sonic <https://maggichk.github.io/papers/sonic.pdf>`_.

    A special batch normalization method has been implemented.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True,
                 track_running_stats: bool = True, device=None):
        """
        .. note::
            The weight and bias are initialized as zero tensor, and these parameters are not updated with gradient (requires_grad=False), which indicates that they may not directly participate in the learning process, but rather be shared or adjusted through other mechanisms.

        :param num_features: The number of input features, which is usually equivalent to the number of channels in the input data.
        :type num_features: int
        :param eps: A small value used to prevent division by zero (The default value is 1e-5.)
        :type eps: float
        :param momentum: The momentum used to update the running mean and variance.(The default value is 0.1).
        :type momentum: float
        :param affine: Whether to use learnable affine transformations
        :type affine: bool
        :param track_running_stats: Whether to track runtime statistics (mean and variance)
        :type track_running_stats: bool
        :param device: Specify the device where the parameter is stored
        :type device: str

        ATTRIBUTES:
            * **weight** (*torch.Tensor*): The weights of neural networks.
            * **bias** (*torch.Tensor*): The bias of neural network

        """
        super(SecBatchNorm2d, self).__init__()
        self.weight = torch.nn.Parameter(torch.zeros([num_features], dtype=data_type), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros([num_features], dtype=data_type), requires_grad=False)

    def forward(self, x):
        """
        Forward propagation function.

        Set ``epsilon1`` and ``epsilon2`` to share using
        :func:`~NssMPC.application.neural_network.functional.functional.torch2share`, then scale ``x`` with
        ``epsilon1`` and pan the result with ``epsilon2``.

        :param x: The data tensor to be processed
        :type x: ArithmeticSecretSharing or ReplicatedSecretSharing
        :return: Obfuscate the processed data tensor
        :rtype: torch.Tensor
        """
        epsilon1 = torch2share(self.weight, x.__class__, x.dtype).unsqueeze(1).unsqueeze(2)
        epsilon2 = torch2share(self.bias, x.__class__, x.dtype).unsqueeze(1).unsqueeze(2)
        return (x * epsilon1) + epsilon2
