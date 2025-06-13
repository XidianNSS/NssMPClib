#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch
from NssMPC.application.neural_network.functional.functional import torch2share
from NssMPC.config import DEVICE


class SecLinear(torch.nn.Module):
    """
    * The implementation of this class is mainly based on the `paper Sonic <https://maggichk.github.io/papers/sonic.pdf>`_.

    The class is used to perform linear transformation operations.
    """

    def __init__(self, input, output, weight=None, bias=None, device=DEVICE):
        """
        Example Initialize the SecLinear.

        Define ``self.weight`` and ``self.bias`` as model parameters, initialized to zero, and with a data type of int64. The ``requires_grad`` is set to **False**, which means that they will not be updated during backpropagation.

        .. note::
            If you want to make the weights and biases trainable, you can set ``requires_grad`` to **True**.

        :param input: Dimension of the input feature
        :type input: int
        :param output: Dimension of the output feature
        :type output: int
        :param weight: Optional weight parameters, (default is **None**).
        :type weight: torch.Tensor
        :param bias: Optional bias parameters, (default is **None**).
        :type bias: torch.Tensor
        :param device: The device where tensors are stored.
        :type device: str

         ATTRIBUTES:
            * **device** (*str*): The device where tensors are stored.
            * **weight** (*torch.Tensor*): The weights of neural networks.
            * **bias** (*torch.Tensor*): bias term

        """
        super(SecLinear, self).__init__()
        self.device = device
        self.weight = torch.nn.Parameter(torch.zeros([output, input], dtype=torch.int64), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros([output], dtype=torch.int64), requires_grad=False)

    def forward(self, x):
        """
        The forward propagation process.

        The weights and biases are first converted to shared form using the function :func:`~NssMPC.application.neural_network.functional.functional.torch2share`, and then ``z`` is computed by matrix multiplication and addition.

        :param x: Input tensor, typically coming from the output of the previous layer.
        :type x: ArithmeticSecretSharing
        :return: The returned result ``z`` is calculated by a linear transformation.
        :rtype: ArithmeticSecretSharing
        """
        weight = torch2share(self.weight, x.__class__, x.dtype).T
        bias = torch2share(self.bias, x.__class__, x.dtype)
        z = (x @ weight) + bias
        return z
