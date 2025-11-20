"""
This document defines different pooling methods: maximum pooling, average pooling, and adaptive average pooling.
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import math

import torch
from NssMPC.infra.tensor import RingTensor

from NssMPC.application.neural_network.functional.functional import img2col_for_pool

from NssMPC.config import DEVICE, data_type


class SecMaxPool2d(torch.nn.Module):
    """
    Max Pooling Layer.

    The implementation of this class is mainly based on the paper Sonic.
    """

    def __init__(self, kernel_size, stride, padding=(0, 0), device=DEVICE):
        """
        Initializes SecMaxPool2d.

        Args:
            kernel_size (int or tuple): Shape of the convolution kernel.
            stride (int or tuple): Convolution stride.
            padding (int or tuple, optional): Input boundary padding.
            device (str, optional): The device where tensors are stored.

        Examples:
            >>> pool = SecMaxPool2d(kernel_size=2, stride=2)
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
        Calculates the output shape based on input shape.

        Args:
            x (RingTensor): Input tensor.

        Returns:
            tuple: The output tensor shape (n, img_c, out_h, out_w).

        Examples:
            >>> shape = pool.get_out_shape(input_tensor)
        """
        n, img_c, img_h, img_w = x.shape
        out_h = math.ceil((img_h - self.kernel_shape + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_shape + 2 * self.padding) // self.stride) + 1

        return n, img_c, out_h, out_w

    def forward(self, x):
        """
        Performs max pooling on the input.

        Args:
            x (RingTensor): Input tensor.

        Returns:
            RingTensor: Tensor after maximum pooling treatment.

        Examples:
            >>> output = pool(input_tensor)
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
    Average Pooling Layer.
    """

    def __init__(self, kernel_size, stride=None, padding=(0, 0), device=DEVICE):
        """
        Initializes SecAvgPool2d.

        Args:
            kernel_size (int or tuple): Shape of the convolution kernel.
            stride (int or tuple, optional): Convolution stride.
            padding (int or tuple, optional): Input boundary padding.
            device (str, optional): The device where tensors are stored.

        Examples:
            >>> pool = SecAvgPool2d(kernel_size=2, stride=2)
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
        Calculates the output shape based on input shape.

        Args:
            x (RingTensor): Input tensor.

        Returns:
            tuple: The output tensor shape (n, img_c, out_h, out_w).

        Examples:
            >>> shape = pool.get_out_shape(input_tensor)
        """
        n, img_c, img_h, img_w = x.shape
        out_h = math.ceil((img_h - self.kernel_size + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_size + 2 * self.padding) // self.stride) + 1
        return n, img_c, out_h, out_w

    def forward(self, x):
        """
        Performs average pooling on the input.

        Args:
            x (RingTensor): Input tensor.

        Returns:
            RingTensor: The tensor after average pooling.

        Examples:
            >>> output = pool(input_tensor)
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
    Adaptive Average Pooling Layer.

    Note:
        SecAdaptiveAvgPool2d automatically calculates the desired convolution kernel size and stride length
        to ensure that the output reaches the specified shape.
    """

    def __init__(self, output_size):
        """
        Initializes SecAdaptiveAvgPool2d.

        Args:
            output_size (int or tuple): Shape of the output.

        Examples:
            >>> pool = SecAdaptiveAvgPool2d(output_size=(5, 5))
        """
        super(SecAdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        """
        Performs adaptive average pooling on the input.

        Args:
            x (RingTensor): Input tensor.

        Returns:
            RingTensor: The tensor after adaptive average pooling.

        Examples:
            >>> output = pool(input_tensor)
        """
        # Transform AdaptiveAvgPool to AvgPool
        input_shape = torch.tensor(x.shape[2:])
        output_shape = torch.tensor(self.output_size)

        stride = torch.floor(input_shape / output_shape).to(data_type)

        kernel_size = input_shape - (output_shape - 1) * stride

        avg_pool = SecAvgPool2d(kernel_size=kernel_size.tolist(), stride=stride.tolist(), padding=0)
        return avg_pool(x)
