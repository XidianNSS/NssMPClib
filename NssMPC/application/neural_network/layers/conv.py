"""
The SecConv2d class is a convolutional layer tailored to specific needs, with a particular focus on data privacy
and security. It ensures data protection in multi-party computing scenarios by controlling how weights and bias items
are shared. For reference, see the `paper <https://maggichk.github.io/papers/sonic.pdf>`_.
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import math
import torch
from NssMPC.application.neural_network.functional.functional import img2col_for_conv, torch2share
from NssMPC.config import DEVICE


class SecConv2d(torch.nn.Module):
    """
    A secure convolution operation suitable for scenarios where data privacy and sharing are required.

    The implementation of this class is mainly based on the paper Sonic.
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), padding=(0, 0), dilation=1, bias=None,
                 device=DEVICE):
        """
        Initializes the SecConv2d layer.

        Args:
            in_channel (int): The number of input channels.
            out_channel (int): The number of output channels.
            kernel_size (int): The size of the convolutional kernel.
            stride (tuple or list or int, optional): Convolution stride. Defaults to (1, 1).
            padding (tuple or list, optional): Input boundary padding. Defaults to (0, 0).
            dilation (int, optional): The spacing between the elements of the convolutional kernel. Defaults to 1.
            bias (torch.Tensor, optional): Bias term.
            device (str, optional): The device where tensors are stored.

        Examples:
            >>> conv = SecConv2d(in_channel=3, out_channel=16, kernel_size=3)
        """
        # TODO: support dilation and different padding, stride for height and width
        super(SecConv2d, self).__init__()
        self.weight = torch.nn.Parameter(
            torch.zeros([out_channel, in_channel, kernel_size, kernel_size], dtype=torch.int64), requires_grad=False)
        self.kernel_shape = None
        if type(stride) in (tuple, list):
            self.stride = stride[0]
        else:
            self.stride = stride
        if type(padding) in (tuple, list):
            padding = padding[0]
        else:
            padding = padding
        self.padding = padding
        self.dilation = 1
        self.out_shape = None
        self.bias = torch.nn.Parameter(torch.zeros(out_channel, dtype=torch.int64), requires_grad=False)
        self.device = device

    def get_out_shape(self, x):
        """
        Calculates the shape of the output.

        Args:
            x (RingTensor): The input tensor.

        Returns:
            tuple: The batch size, the number of output channels, height, and width.

        Examples:
            >>> shape = conv.get_out_shape(input_tensor)
        """
        n, img_c, img_h, img_w = x.shape
        kn, kc, kh, kw = self.kernel_shape
        out_h = math.ceil((img_h - self.kernel_shape[2] + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_shape[2] + 2 * self.padding) // self.stride) + 1

        return n, kn, out_h, out_w

    def forward(self, x):
        """
        Defines the convolution forward propagation process.

        Args:
            x (ArithmeticSecretSharing or ReplicatedSecretSharing): The input tensor.

        Returns:
            RingTensor: The output tensor after the convolution operation.

        Examples:
            >>> output = conv(input_tensor)
        """
        weight = torch2share(self.weight, x.__class__, x.dtype)

        self.kernel_shape = weight.shape
        self.out_shape = self.get_out_shape(x)

        padding = self.padding
        x = x.pad((padding, padding, padding, padding), mode="constant", value=0)

        kN, kC, ksize, _ = self.kernel_shape

        x = img2col_for_conv(x, ksize, self.stride).transpose(1, 2)

        weight = weight.reshape((kN, kC * ksize * ksize))
        weight = weight.T

        output = x @ weight

        bias = self.bias

        if bias is None:
            pass
        else:
            bias = torch2share(self.bias, x.__class__, x.dtype)
            output = output + bias

        return output.transpose(1, 2).reshape(self.out_shape)
