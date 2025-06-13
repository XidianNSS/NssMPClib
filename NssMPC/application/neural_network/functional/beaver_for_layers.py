"""
Generate the required beaver according to the needs of the neural network layers
Support matrix beaver for convolutional, linear layers and average pooling layers
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import warnings
import torch

from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.aux_parameter import MatmulTriples, RssMatmulTriples
from NssMPC.secure_model.mpc_party import SemiHonestCS, HonestMajorityParty


def beaver_for_conv(x, kernel, padding, stride, num_of_triples):
    """
    The way to generate matrix beaver triples of convolutional layers.

    First calculate the height and width of the feature map, and then determine the type of ``x.party``:
        * If x is :class:`~NssMPC.secure_model.mpc_party.semi_honest.SemiHonestCS` (Semi-honest Computation Model): Call the :meth:`~NssMPC/crypto/aux_parameter/beaver_triples/arithmetic_triples.MatmulTriples.gen` method to generate the multiplication triple.
        * If x is :class:`~NssMPC.secure_model.mpc_party.honest_majority.HonestMajorityParty` (Honest Majority Model): Call the :meth:`~NssMPC/crypto/aux_parameter/beaver_triples/arithmetic_triples.RssMatmulTriples.gen` method to generate the multiplication triple
        * If the party type is not among the known categories: A warning is issued and the default call to :meth:`~NssMPC/crypto/aux_parameter/beaver_triples/arithmetic_triples.RssMatmulTriples.gen` is made.

    :param x: the input tensor
    :type x: torch.Tensor
    :param kernel: the convolutional kernel
    :type kernel: torch.Tensor
    :param padding: Fill in the data at the periphery edge
    :type padding: int
    :param stride: The step size of the position at each slide
    :type stride: int
    :param num_of_triples: the number of triples
    :type num_of_triples: int
    """
    n, c, h, w = x.shape
    f, _, k, _ = kernel.shape
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w + 2 * padding - k) // stride + 1
    im2col_output_shape = torch.zeros([n, h_out * w_out, c * k * k]).shape
    reshaped_kernel_size = torch.zeros([1, c * k * k, f]).shape
    shapes = im2col_output_shape, reshaped_kernel_size
    if isinstance(PartyRuntime.party, SemiHonestCS):
        return MatmulTriples.gen(num_of_triples, shapes[0], shapes[1])
    elif isinstance(PartyRuntime.party, HonestMajorityParty):
        return RssMatmulTriples.gen(num_of_triples, shapes[0], shapes[1])
    else:
        warnings.warn("Maybe this party do not need to generate beaver triples.")
        return RssMatmulTriples.gen(num_of_triples, shapes[0], shapes[1])


def beaver_for_linear(x, weight, num_of_triples):
    """
    The way to generate matrix beaver triples of linear layers.

    First transpose the weight so that it matches the shape of the input tensor. Then determine what type of triples to generate based on the type of ``x.party``:
        * If x is :class:`~NssMPC.secure_model.mpc_party.semi_honest.SemiHonestCS` (Semi-honest Computation Model): Call the :meth:`~NssMPC/crypto/aux_parameter/beaver_triples/arithmetic_triples.MatmulTriples.gen` method to generate the multiplication triple
        * If x is :class:`~NssMPC.secure_model.mpc_party.honest_majority.HonestMajorityParty` (Honest Majority Model): Call the :meth:`~NssMPC/crypto/aux_parameter/beaver_triples/arithmetic_triples.RssMatmulTriples.gen` method to generate the multiplication triple
        * If the party type is not among the known categories: A warning is issued and the default call to :meth:`~NssMPC/crypto/aux_parameter/beaver_triples/arithmetic_triples.RssMatmulTriples.gen` is made.

    :param x: The input tensor
    :type x: torch.Tensor
    :param weight: Multiply each element of an image
    :type weight: torch.Tensor
    :param num_of_triples: the number of triples
    :type num_of_triples: int
    """

    weight = weight.T
    if isinstance(PartyRuntime.party, SemiHonestCS):
        return MatmulTriples.gen(num_of_triples, x.shape, weight.shape)
    elif isinstance(PartyRuntime.party, HonestMajorityParty):
        return RssMatmulTriples.gen(num_of_triples, x.shape, weight.shape)
    else:
        warnings.warn("Maybe this party do not need to generate beaver triples.")
        return RssMatmulTriples.gen(num_of_triples, x.shape, weight.shape)


def beaver_for_avg_pooling(x, kernel_shape, padding, stride, num_of_triples):
    """
    The way to generate matrix beaver triples of average pooling layers.The function logic is basically the same as :func:`beaver_for_conv`.

    :param x: the input tensor
    :type x: torch.Tensor
    :param kernel_shape: The width or height of the convolution kernel
    :type kernel_shape: int
    :param padding: Fill in the data at the periphery edge
    :type padding: int
    :param stride: The step size of the position at each slide
    :type stride: int
    :param num_of_triples: the number of triples
    :type num_of_triples: int
    """
    n, c, h, w = x.shape
    h_out = (h + 2 * padding - kernel_shape) // stride + 1
    w_out = (w + 2 * padding - kernel_shape) // stride + 1
    shapes = torch.zeros([n, c, h_out * w_out, kernel_shape * kernel_shape]).shape
    if isinstance(PartyRuntime.party, SemiHonestCS):
        return MatmulTriples.gen(num_of_triples, shapes, torch.zeros([shapes[3], 1]).shape)
    elif isinstance(PartyRuntime.party, HonestMajorityParty):
        return RssMatmulTriples.gen(num_of_triples, shapes, torch.zeros([shapes[3], 1]).shape)
    else:
        warnings.warn("Maybe this party do not need to generate beaver triples.")
        return RssMatmulTriples.gen(num_of_triples, shapes, torch.zeros([shapes[3], 1]).shape)


def beaver_for_adaptive_avg_pooling(x, output_shape, num_of_triples):
    """
    The way to generate matrix beaver triples of adaptive average pooling layers.

    After obtaining the shape of the input and output tensors, calculate the step length(``stride``) and convolution kernel size(``kernel_size``),
    convert ``kernel_size`` and ``stride`` into a Python list, which is convenient to pass to the function later,
    and then call the function :func:`beaver_for_avg_pooling` to generate the Beaver triplet.

    :param x: The input tensor
    :type x: torch.Tensor
    :param output_shape: the shape of the output tensor
    :type output_shape: tuple
    :param num_of_triples: the number of triples
    :type num_of_triples: int
    """
    input_shape = torch.tensor(x.shape[2:])
    output_shape = torch.tensor(output_shape)

    stride = torch.floor(input_shape / output_shape).to(torch.int64)
    kernel_size = input_shape - (output_shape - 1) * stride

    kernel_size_list = kernel_size.tolist()
    stride_list = stride.tolist()

    return beaver_for_avg_pooling(x, kernel_shape=kernel_size_list[0], padding=0, stride=stride_list[0],
                                  num_of_triples=num_of_triples)
