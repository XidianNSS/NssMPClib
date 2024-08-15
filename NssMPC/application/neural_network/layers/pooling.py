import math

import torch
from NssMPC.common.ring.ring_tensor import RingTensor

from NssMPC.application.neural_network.functional.functional import img2col_for_pool

from NssMPC.config import DEVICE, data_type


class SecMaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride, padding=(0, 0), device=DEVICE):
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
        n, img_c, img_h, img_w = x.shape
        out_h = math.ceil((img_h - self.kernel_shape + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_shape + 2 * self.padding) // self.stride) + 1

        return n, img_c, out_h, out_w

    def forward(self, x):
        k_size = self.kernel_shape
        out_shape = self.get_out_shape(x)

        padding = self.padding
        x = x.pad((padding, padding, padding, padding), mode="constant", value=0)

        x = img2col_for_pool(x, k_size, self.stride)
        xs = x.__class__.max(x, dim=-2)

        return xs.reshape(out_shape)


class SecAvgPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=(0, 0), device=DEVICE):
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
        n, img_c, img_h, img_w = x.shape
        out_h = math.ceil((img_h - self.kernel_size + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_size + 2 * self.padding) // self.stride) + 1
        return n, img_c, out_h, out_w

    def forward(self, x):
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
    def __init__(self, output_size):
        super(SecAdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        # Transform AdaptiveAvgPool to AvgPool
        input_shape = torch.tensor(x.shape[2:])
        output_shape = torch.tensor(self.output_size)

        stride = torch.floor(input_shape / output_shape).to(data_type)

        kernel_size = input_shape - (output_shape - 1) * stride

        avg_pool = SecAvgPool2d(kernel_size=kernel_size.tolist(), stride=stride.tolist(), padding=0)
        return avg_pool(x)
