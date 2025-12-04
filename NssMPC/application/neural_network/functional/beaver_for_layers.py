"""
Generate the required beaver according to the needs of the neural communication layers
Support matrix beaver for convolutional, linear layers and average pooling layers
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import warnings

import torch

from NssMPC import Party2PC, Party3PC, SEMI_HONEST, HONEST_MAJORITY
from NssMPC.infra.mpc.party import PartyCtx, Party
from NssMPC.protocols.honest_majority_3pc.multiplication import RssMatmulTriples
from NssMPC.protocols.semi_honest_2pc.multiplication import MatmulTriples


def beaver_for_conv(x, kernel, padding, stride, num_of_triples, party: Party = PartyCtx.get()):
    """Generate matrix Beaver triples for convolutional layers.

    This function calculates the height and width of the feature map, and dispatches the generation task
    based on the type of ``x.party``:

    * If **Semi-Honest** (:class:`~NssMPC.runtime.party.semi_honest.SemiHonestCS`):
      Calls :meth:`~NssMPC.crypto.aux_parameter.beaver_triples.arithmetic_triples.MatmulTriples.gen`.
    * If **Honest-Majority** (:class:`~NssMPC.runtime.party.honest_majority.HonestMajorityParty`):
      Calls :meth:`~NssMPC.crypto.aux_parameter.beaver_triples.arithmetic_triples.RssMatmulTriples.gen`.
    * **Fallback**: Issues a warning and defaults to ``RssMatmulTriples.gen``.

    Args:
        x (torch.Tensor or RingTensor): The input tensor, used to determine shapes and the associated party.
        kernel (torch.Tensor): The convolutional kernel tensor.
        padding (int): The amount of implicit padding on both sides.
        stride (int): The stride of the convolving kernel.
        num_of_triples (int): The number of triples to generate.
        party: The party instance. Defaults to the current context specific party.

    Returns:
        list: A list or collection of generated multiplication triples.

    Examples:
        >>> triples = beaver_for_conv(x, kernel, padding=1, stride=1, num_of_triples=10)
    """
    n, c, h, w = x.shape
    f, _, k, _ = kernel.shape
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w + 2 * padding - k) // stride + 1
    im2col_output_shape = torch.zeros([n, h_out * w_out, c * k * k]).shape
    reshaped_kernel_size = torch.zeros([1, c * k * k, f]).shape
    shapes = im2col_output_shape, reshaped_kernel_size
    if isinstance(party, Party2PC):
        return MatmulTriples.gen(num_of_triples, shapes[0], shapes[1])
    elif isinstance(party, Party3PC) and party.thread_model_cfg == HONEST_MAJORITY:
        return RssMatmulTriples.gen(num_of_triples, shapes[0], shapes[1])
    else:
        warnings.warn("Maybe this party do not need to generate beaver triples.")
        return RssMatmulTriples.gen(num_of_triples, shapes[0], shapes[1])


def beaver_for_linear(x, weight, num_of_triples, party: Party = PartyCtx.get()):
    """Generate matrix Beaver triples for linear (fully connected) layers.

    First, the weight tensor is transposed to match the input tensor's shape for matrix multiplication.
    Then, the generation strategy is dispatched based on the type of ``x.party``:

    * **Semi-Honest** (:class:`~NssMPC.runtime.party.semi_honest.SemiHonestCS`):
      Calls :meth:`~NssMPC.crypto.aux_parameter.beaver_triples.arithmetic_triples.MatmulTriples.gen`.
    * **Honest-Majority** (:class:`~NssMPC.runtime.party.honest_majority.HonestMajorityParty`):
      Calls :meth:`~NssMPC.crypto.aux_parameter.beaver_triples.arithmetic_triples.RssMatmulTriples.gen`.
    * **Fallback**: Issues a warning and defaults to ``RssMatmulTriples.gen``.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor of the linear layer.
        num_of_triples (int): The number of triples to generate.
        party: The party instance. Defaults to the current context specific party.

    Returns:
        list: A list or collection of generated multiplication triples.

    Examples:
        >>> triples = beaver_for_linear(x, weight, num_of_triples=10)
    """

    weight = weight.T
    if isinstance(party, Party2PC):
        return MatmulTriples.gen(num_of_triples, x.shape, weight.shape)
    elif isinstance(party, Party3PC) and party.thread_model_cfg == HONEST_MAJORITY:
        return RssMatmulTriples.gen(num_of_triples, x.shape, weight.shape)
    else:
        warnings.warn("Maybe this party do not need to generate beaver triples.")
        return RssMatmulTriples.gen(num_of_triples, x.shape, weight.shape)


def beaver_for_avg_pooling(x, kernel_shape, padding, stride, num_of_triples, party: Party = PartyCtx.get()):
    """Generate matrix Beaver triples for average pooling layers.

    The function logic follows the same dispatch mechanism as :func:`beaver_for_conv`.
    It prepares the necessary pre-computed data (triples) to compute average pooling via matrix operations or secure division.

    Args:
        x (torch.Tensor): The input tensor.
        kernel_shape (int): The size of the pooling window (kernel size).
        padding (int): The amount of implicit padding on both sides.
        stride (int): The stride of the pooling window.
        num_of_triples (int): The number of triples to generate.
        party: The party instance. Defaults to the current context specific party.

    Returns:
        list: A list or collection of generated multiplication triples.

    Examples:
        >>> triples = beaver_for_avg_pooling(x, kernel_shape=2, padding=0, stride=2, num_of_triples=10)
    """
    n, c, h, w = x.shape
    h_out = (h + 2 * padding - kernel_shape) // stride + 1
    w_out = (w + 2 * padding - kernel_shape) // stride + 1
    shapes = torch.zeros([n, c, h_out * w_out, kernel_shape * kernel_shape]).shape
    if isinstance(party, Party2PC):
        return MatmulTriples.gen(num_of_triples, shapes, torch.zeros([shapes[3], 1]).shape)
    elif isinstance(party, Party3PC) and party.thread_model_cfg == HONEST_MAJORITY:
        return RssMatmulTriples.gen(num_of_triples, shapes, torch.zeros([shapes[3], 1]).shape)
    else:
        warnings.warn("Maybe this party do not need to generate beaver triples.")
        return RssMatmulTriples.gen(num_of_triples, shapes, torch.zeros([shapes[3], 1]).shape)


def beaver_for_adaptive_avg_pooling(x, output_shape, num_of_triples):
    """Generate matrix beaver triples of adaptive average pooling layers.

    After obtaining the shape of the input and output tensors, calculate the step length(``stride``) and convolution kernel size(``kernel_size``),
    convert ``kernel_size`` and ``stride`` into a Python list, which is convenient to pass to the function later,
    and then call the function :func:`beaver_for_avg_pooling` to generate the Beaver triplet.

    Args:
        x (torch.Tensor): The input tensor.
        output_shape (tuple): The shape of the output tensor.
        num_of_triples (int): The number of triples.

    Returns:
        list: A list or collection of generated multiplication triples.

    Examples:
        >>> triples = beaver_for_adaptive_avg_pooling(x, output_shape=(5, 5), num_of_triples=10)
    """
    input_shape = torch.tensor(x.shape[2:])
    output_shape = torch.tensor(output_shape)

    stride = torch.floor(input_shape / output_shape).to(torch.int64)
    kernel_size = input_shape - (output_shape - 1) * stride

    kernel_size_list = kernel_size.tolist()
    stride_list = stride.tolist()

    return beaver_for_avg_pooling(x, kernel_shape=kernel_size_list[0], padding=0, stride=stride_list[0],
                                  num_of_triples=num_of_triples)
