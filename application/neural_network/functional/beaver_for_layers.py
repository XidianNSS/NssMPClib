"""
Generate the required beaver according to the needs of the neural network layers
Support matrix beaver for convolutional and linear layers
"""

import torch

from crypto.primitives.beaver.matrix_triples import MatrixTriples


def beaver_for_conv(x, kernel, padding, stride, num_of_triples):
    """
    The way to generate matrix beaver triples of convolutional layers

    Args:
        x: the input tensor
        kernel: the convolutional kernel
        padding
        stride
        num_of_triples: the number of triples
    """
    n, c, h, w = x.shape
    f, _, k, _ = kernel.shape
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w + 2 * padding - k) // stride + 1
    im2col_output_shape = torch.zeros([n, h_out * w_out, c * k * k]).shape
    reshaped_kernel_size = torch.zeros([1, c * k * k, f]).shape
    shapes = im2col_output_shape, reshaped_kernel_size
    return MatrixTriples.gen(num_of_triples, shapes[0], shapes[1])


def beaver_for_linear(x, weight, num_of_triples):
    """
    The way to generate matrix beaver triples of linear layers

    Args:
        x: the input tensor
        weight
        num_of_triples: the number of triples
    """

    weight = weight.T
    return MatrixTriples.gen(num_of_triples, x.shape, weight.shape)


def beaver_for_avg_pooling(x, kernel_shape, padding, stride, num_of_triples):
    """
    The way to generate matrix beaver triples of average pooling layers

    Args:
        x: the input tensor
        kernel_shape
        padding
        stride
        num_of_triples: the number of triples
    """
    n, c, h, w = x.shape
    h_out = (h + 2 * padding - kernel_shape) // stride + 1
    w_out = (w + 2 * padding - kernel_shape) // stride + 1
    shapes = torch.zeros([n, c, h_out * w_out, kernel_shape * kernel_shape]).shape
    return MatrixTriples.gen(num_of_triples, shapes, torch.zeros([shapes[3], 1]).shape)


def beaver_for_adaptive_avg_pooling(x, output_shape, num_of_triples):
    """
    The way to generate matrix beaver triples of adaptive average pooling layers

    Args:
        x: the input tensor
        output_shape: the shape of the output tensor
        num_of_triples: the number of triples
    """
    input_shape = torch.tensor(x.shape[2:])
    output_shape = torch.tensor(output_shape)

    stride = torch.floor(input_shape / output_shape).to(torch.int64)
    kernel_size = input_shape - (output_shape - 1) * stride

    kernel_size_list = kernel_size.tolist()
    stride_list = stride.tolist()

    return beaver_for_avg_pooling(x, kernel_shape=kernel_size_list[0], padding=0, stride=stride_list[0],
                                  num_of_triples=num_of_triples)
