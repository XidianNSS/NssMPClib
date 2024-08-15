import math
import torch
from NssMPC.application.neural_network.functional.functional import img2col_for_conv, torch2share
from NssMPC.config import DEVICE


class SecConv2d(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), padding=(0, 0), dilation=1, bias=None,
                 device=DEVICE):
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
        n, img_c, img_h, img_w = x.shape
        kn, kc, kh, kw = self.kernel_shape
        out_h = math.ceil((img_h - self.kernel_shape[2] + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_shape[2] + 2 * self.padding) // self.stride) + 1

        return n, kn, out_h, out_w

    def forward(self, x):
        weight = torch2share(self.weight, x.__class__, x.dtype, x.party)

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
            bias = torch2share(self.bias, x.__class__, x.dtype, x.party)
            output = output + bias

        return output.transpose(1, 2).reshape(self.out_shape)
