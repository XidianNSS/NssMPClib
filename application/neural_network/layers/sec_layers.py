from application.neural_network.functional.functional import *
from common.tensor import *
from crypto.primitives.arithmetic_secret_sharing.arithmetic_shared_ring_tensor import ArithmeticSharedRingTensor


class SecConv2d(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), padding=(0, 0), bias=None, device=DEVICE):
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
        self.out_shape = None
        self.bias = torch.nn.Parameter(torch.zeros(out_channel, dtype=torch.int64), requires_grad=False)
        self.device = device

    def get_out_shape(self, x: ArithmeticSharedRingTensor):
        n, img_c, img_h, img_w = x.shape
        kn, kc, kh, kw = self.kernel_shape
        out_h = math.ceil((img_h - self.kernel_shape[2] + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_shape[2] + 2 * self.padding) // self.stride) + 1

        return n, kn, out_h, out_w

    def forward(self, x: ArithmeticSharedRingTensor):
        self.kernel_shape = self.weight.shape
        self.out_shape = self.get_out_shape(x)

        padding = self.padding
        x = x.pad((padding, padding, padding, padding), mode="constant", value=0)

        weight = ArithmeticSharedRingTensor(RingTensor(self.weight, x.ring_tensor.dtype), x.party)

        kN, kC, ksize, _ = self.kernel_shape

        x = img2col_for_conv(x, ksize, self.stride).transpose(1, 2)

        weight = weight.reshape((kN, kC * ksize * ksize))
        weight = weight.T

        output = x @ weight

        bias = self.bias

        if bias is None:
            pass
        else:
            bias = ArithmeticSharedRingTensor(RingTensor(self.bias, x.ring_tensor.dtype), x.party)
            output = output + bias

        return output.transpose(1, 2).reshape(self.out_shape)


class SecLinear(torch.nn.Module):
    def __init__(self, input, output, weight=None, bias=None, device=DEVICE):
        super(SecLinear, self).__init__()
        self.device = device
        self.weight = torch.nn.Parameter(torch.zeros([output, input], dtype=torch.int64), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros([output], dtype=torch.int64), requires_grad=False)

    def forward(self, x):
        weight = ArithmeticSharedRingTensor(RingTensor(self.weight.T, x.ring_tensor.dtype), x.party)
        bias = ArithmeticSharedRingTensor(RingTensor(self.bias, x.ring_tensor.dtype), x.party)
        z = (x @ weight) + bias
        return z


class SecReLU(torch.nn.Module):
    def __init__(self, inplace=True):
        super(SecReLU, self).__init__()

    def forward(self, x: ArithmeticSharedRingTensor):
        return (x > 0) * x


def _SecReLU(x):
    return SecReLU()(x)


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

    def get_out_shape(self, x: ArithmeticSharedRingTensor):
        n, img_c, img_h, img_w = x.shape
        out_h = math.ceil((img_h - self.kernel_shape + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_shape + 2 * self.padding) // self.stride) + 1

        return n, img_c, out_h, out_w

    def forward(self, x: ArithmeticSharedRingTensor):
        k_size = self.kernel_shape
        out_shape = self.get_out_shape(x)

        padding = self.padding
        x = x.pad((padding, padding, padding, padding), mode="constant", value=0)

        x = img2col_for_pool(x, k_size, self.stride)
        xs = ArithmeticSharedRingTensor.max(x, dim=-2)

        return xs.reshape(out_shape).to(self.device)


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

    def get_out_shape(self, x: ArithmeticSharedRingTensor):
        n, img_c, img_h, img_w = x.shape
        out_h = math.ceil((img_h - self.kernel_size + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_size + 2 * self.padding) // self.stride) + 1
        return n, img_c, out_h, out_w

    def forward(self, x: ArithmeticSharedRingTensor):
        padding = self.padding
        x = x.pad((padding, padding, padding, padding), mode="constant", value=0)

        k_size = self.kernel_size
        out_shape = self.get_out_shape(x)

        x = img2col_for_pool(x, k_size, self.stride).transpose(2, 3)

        # Average pooling is equivalent to performing a convolution with weights of 1/n.
        weight = RingFunc.ones((x.shape[3], 1), x.dtype) // (k_size * k_size)

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


class SecBatchNorm2d(torch.nn.Module):
    def __init__(self, channel):
        super(SecBatchNorm2d, self).__init__()
        self.weight = torch.nn.Parameter(torch.zeros([channel], dtype=data_type), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros([channel], dtype=data_type), requires_grad=False)

    def forward(self, x):
        epsilon1 = ArithmeticSharedRingTensor(
            RingTensor(self.weight.unsqueeze(1).unsqueeze(2), x.ring_tensor.dtype), x.party)
        epsilon2 = ArithmeticSharedRingTensor(
            RingTensor(self.bias.unsqueeze(1).unsqueeze(2), x.ring_tensor.dtype), x.party)
        return (x * epsilon1) + epsilon2
