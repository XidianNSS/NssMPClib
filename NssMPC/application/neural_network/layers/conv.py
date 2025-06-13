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
    * The implementation of this class is mainly based on the `paper Sonic <https://maggichk.github.io/papers/sonic.pdf>`_.

    A secure convolution operation has been implemented, suitable for scenarios where data privacy and sharing are required.
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), padding=(0, 0), dilation=1, bias=None,
                 device=DEVICE):
        """
        :param in_channel: The number of input channels.
        :type in_channel: int
        :param out_channel: The number of output channels.
        :type out_channel: int
        :param kernel_size: The size of the convolutional kernel.
        :type kernel_size: int
        :param stride: Convolution stride (the default value is **(1, 1)**).
        :type stride: tuple or list or int
        :param padding: Input boundary padding (default is **(0, 0)**).
        :type padding: tuple or list
        :param dilation: The spacing between the elements of the convolutional kernel (default is **1**).
        :type dilation: int
        :param bias: bias term
        :type bias: torch.Tensor
        :param device: The device where tensors are stored.
        :type device: str

        ATTRIBUTES:
            * **weight** (*torch.Tensor*): The weights of neural networks
            * **kernel_shape** (*torch.Tensor*): Shape of the convolution kernel
            * **stride** (*int*): Convolution stride
            * **padding** (*int*): Input boundary padding
            * **dilation** (*int*): In order to make the CNN model can capture longer distances without increasing the model parameters.
            * **out_shape** (*torch.Tensor*): Shape of the output data
            * **bias** (*torch.Tensor*): bias term
            * **device** (*str*): The device where tensors are stored

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
        Calculate the shape of the output.

        The output height and width are calculated based on the input height, width, size of the convolutional kernel, padding, and stride.

        :param x: The input tensor
        :type x: RingTensor
        :return: The batch size, the number of output channels, height, and width.
        :rtype: tuple
        """
        n, img_c, img_h, img_w = x.shape
        kn, kc, kh, kw = self.kernel_shape
        out_h = math.ceil((img_h - self.kernel_shape[2] + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_shape[2] + 2 * self.padding) // self.stride) + 1

        return n, kn, out_h, out_w

    def forward(self, x):
        """
        Define the convolution forward propagation process.

        The convolution operation means that the convolution kernel slides over the input image and performs
        element-level product and sum at each position to generate an output value. This process is repeated many
        times to generate a complete feature map, and the dimension of the output feature map is equal to the number
        of convolution cores.

        To speed up parallelization, the shape of input ``x`` is changed first, matrix multiplication
        with weight, finally add bias as the result.

        :param x: The input tensor
        :type x: ArithmeticSecretSharing or ReplicatedSecretSharing
        :return: The output tensor after the convolution operation.
        :rtype: RingTensor
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
