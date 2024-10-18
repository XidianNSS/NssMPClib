"""
This document defines different pooling methods: maximum pooling, average pooling, and adaptive average pooling.
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import math

import torch
from NssMPC.common.ring.ring_tensor import RingTensor

from NssMPC.application.neural_network.functional.functional import img2col_for_pool

from NssMPC.config import DEVICE, data_type


class SecMaxPool2d(torch.nn.Module):
    """
    Max Pooling.

    * The implementation of this class is mainly based on the `paper Sonic <https://maggichk.github.io/papers/sonic.pdf>`_.
    """

    def __init__(self, kernel_size, stride, padding=(0, 0), device=DEVICE):
        """
        Example Initialize SecMaxPool2d.

        ATTRIBUTES:
            * **kernel_size** (*int*): Shape of the convolution kernel
            * **stride** (*int*): Convolution stride
            * **padding** (*int*): Input boundary padding
            * **device** (*str*): The device where tensors are stored
        """
        super(SecMaxPool2d, self).__init__()
        if type(kernel_size) in (tuple, list):
            self.kernel_shape = kernel_size[0]
        else:
            self.kernel_shape = kernel_size
        if type(stride) in (tuple, list):
            self.stride = stride[0]
        else:
            self.stride = stride
        if type(padding) in (tuple, list):
            self.padding = padding[0]
        else:
            self.padding = padding
        self.device = device

    def get_out_shape(self, x):
        """
        After obtaining the number of batches and channels from the input ``x``, the output height ``out_h`` and output width ``out_w`` are calculated using the integer up function ceil.

        :param x: Input tensor
        :type x: RingTensor
        :return: The output tensor after the pooling operation.
        :rtype: tuple
        """
        n, img_c, img_h, img_w = x.shape
        out_h = math.ceil((img_h - self.kernel_shape + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_shape + 2 * self.padding) // self.stride) + 1

        return n, img_c, out_h, out_w

    def forward(self, x):
        """
        After getting the size of the pooling window, populate ``x`` with the function
        :func:`~NssMPC.common.ring.ring_tensor.RingTensor.pad`. Then call
        :func:`~NssMPC.application.neural_network.functional.functional.img2col_for_pool` to tensor transform ``x``
        into a column form suitable for pooling. Finally, maximum pooling is performed on the last dimension to
        obtain the maximum value of each window.

        :param x: Input tensor
        :type x: RingTensor
        :return: Tensor after maximum pooling treatment
        :rtype: RingTensor

        """
        k_size = self.kernel_shape
        out_shape = self.get_out_shape(x)

        padding = self.padding
        x = x.pad((padding, padding, padding, padding), mode="constant", value=0)

        x = img2col_for_pool(x, k_size, self.stride)
        xs = x.__class__.max(x, dim=-2)

        return xs.reshape(out_shape)


class SecAvgPool2d(torch.nn.Module):
    """
    Average Pooling.
    """

    def __init__(self, kernel_size, stride=None, padding=(0, 0), device=DEVICE):
        """
        Example Initialize SecAvgPool2d.

        ATTRIBUTES:
            * **kernel_size** (*int*): Shape of the convolution kernel
            * **stride** (*int*): Convolution stride
            * **padding** (*int*): Input boundary padding
            * **device** (*str*): The device where tensors are stored

        """
        super(SecAvgPool2d, self).__init__()

        if isinstance(kernel_size, (tuple, list)):
            kernel_size = kernel_size[0]
        self.kernel_size = kernel_size

        if stride is None:
            stride = self.kernel_size
        elif isinstance(stride, (tuple, list)):
            stride = stride[0]
        self.stride = stride

        if isinstance(padding, (tuple, list)):
            padding = padding[0]
        self.padding = padding

        self.device = device

    def get_out_shape(self, x):
        """
        After obtaining the number of batches ``x`` and channels ``img_c`` from the input ``x``, the output height ``out_h`` and output width ``out_w`` are calculated using the integer up function ceil.

        :param x: Input tensor
        :type x: RingTensor
        :return: The output tensor after the pooling operation.
        :rtype: tuple
        """
        n, img_c, img_h, img_w = x.shape
        out_h = math.ceil((img_h - self.kernel_size + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_size + 2 * self.padding) // self.stride) + 1
        return n, img_c, out_h, out_w

    def forward(self, x):
        """
        After the input is edge-filled, a weight tensor is created and the average of each window is obtained using
        matrix multiplication x @ weight.

        :param x: Input tensor
        :type x: RingTensor
        :return: The tensor after average pooling
        :rtype: RingTensor
        """
        padding = self.padding
        x = x.pad((padding, padding, padding, padding), mode="constant", value=0)

        k_size = self.kernel_size
        out_shape = self.get_out_shape(x)

        x = img2col_for_pool(x, k_size, self.stride).transpose(2, 3)

        # Average pooling is equivalent to performing a convolution with weights of 1/n.
        weight = RingTensor.ones((x.shape[3], 1), x.dtype) // (k_size * k_size)

        res = x @ weight

        return res.reshape(out_shape)


class SecAdaptiveAvgPool2d(torch.nn.Module):
    """
    Adaptive Average Pooling.

    .. note::
        The difference between **SecAdaptiveAvgPool2d** and **SecAvgPool2d** is that:
            * **SecAvgPool2d** is generally used for fixed-size inputs, and the output size is also dependent on the size of the input feature map. Moreover, the size of the output depends on the input size, convolution kernel size and stride length, and there is no guarantee that the shape of the output is expected by the user.

            * **SecAdaptiveAvgPool2d** automatically calculates the desired convolution kernel size and stride length to ensure that the output reaches the specified shape. And the size of the output is user-defined, regardless of the size of the input feature map, it can always be adjusted to the specified output size.
    """

    def __init__(self, output_size):
        """
        Example Initialize SecAdaptiveAvgPool2d.

        ATTRIBUTES:
            * **output_size** (*int*): Shape of the output
        """
        super(SecAdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        """
        After obtaining the shape of the input/output tensor, use the ``torch.floor`` function for operation to round
        down to obtain the step length, calculate the pool window size, create the SecAvgPool2d instance and pass in
        the input x for adaptive average pooling operation.

        :param x: Input tensor
        :type x: RingTensor
        :return: The tensor after adaptive average pooling
        :rtype: RingTensor
        """
        # Transform AdaptiveAvgPool to AvgPool
        input_shape = torch.tensor(x.shape[2:])
        output_shape = torch.tensor(self.output_size)

        stride = torch.floor(input_shape / output_shape).to(data_type)

        kernel_size = input_shape - (output_shape - 1) * stride

        avg_pool = SecAvgPool2d(kernel_size=kernel_size.tolist(), stride=stride.tolist(), padding=0)
        return avg_pool(x)
