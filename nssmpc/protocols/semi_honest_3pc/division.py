#  This file is part of the NssMPClib project.
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""Replicated fixed-point division using secret normalization and Goldschmidt.

All secret arithmetic is dispatched through the active runtime, including the
honest-majority comparison and checked multiplication protocols when selected.
"""

import torch

from nssmpc.config import SCALE_BIT
from nssmpc.infra.tensor import RingTensor
from nssmpc.primitives.secret_sharing import ReplicatedSecretSharing


def get_neg_exp2_k(divisor: ReplicatedSecretSharing) -> ReplicatedSecretSharing:
    """Secret factor that brings a positive divisor into [0.5, 1).

    Requires 2**(-SCALE_BIT) <= divisor < 2**SCALE_BIT. Comparisons
    with public powers of two reveal neither the divisor nor its magnitude.

    Args:
        divisor: A positive fixed-point replicated divisor.

    Returns:
        A secret power-of-two normalization factor with the divisor's shape.
    """
    powers = torch.arange(1 - SCALE_BIT, SCALE_BIT, device=divisor.device)
    thresholds = RingTensor.convert_to_ring(torch.pow(2.0, powers))
    steps = RingTensor.convert_to_ring(torch.pow(2.0, -powers - 1))
    greater = divisor.unsqueeze(-1) >= thresholds
    return 2.0 ** (SCALE_BIT - 1) - (greater * steps).sum(dim=-1)


def sh3pc_div(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing) -> ReplicatedSecretSharing:
    """Approximate x/y using the active replicated-sharing protocols.

    Inputs must be fixed point, with 2**(-f) <= abs(y) < 2**f (f=SCALE_BIT).
    The normalized numerator and all products must fit the ring. Division by
    secret zero is undefined; it is not opened or checked in plaintext.
    Accuracy depends on fixed-point precision, especially for small divisors.

    Args:
        x: The fixed-point replicated dividend.
        y: The nonzero fixed-point replicated divisor, broadcastable with x.

    Returns:
        A replicated sharing of the approximate quotient.

    Raises:
        TypeError: If either input is not a fixed-point sharing.
        RuntimeError: If the input shapes cannot be broadcast together.

    Examples:
        >>> quotient = sh3pc_div(x, y)
    """
    if x.dtype != 'float' or y.dtype != 'float':
        raise TypeError("RSS division requires fixed-point ('float') inputs")
    # Validate public shapes before starting any communication.
    torch.broadcast_shapes(x.shape, y.shape)
    sign = 1 - (y < 0) * 2
    positive_y = y * sign
    factor = get_neg_exp2_k(positive_y)
    a = x * factor
    b = positive_y * factor
    # Refine the reciprocal of b on [0.5, 1) using Goldschmidt's method.
    w = b * (-2) + 2.9142
    error = 1 - b * w
    product = error + 1
    for _ in range(3):
        error = error * error
        product = product * (error + 1)
    return a * w * product * sign


def sh3pc_inv(x: ReplicatedSecretSharing) -> ReplicatedSecretSharing:
    """Computes a reciprocal using the active runtime's division protocol.

    Uses the divisor domain and precision constraints documented by sh3pc_div.

    Args:
        x: A nonzero fixed-point replicated input.

    Returns:
        A replicated sharing of the approximate reciprocal.

    Examples:
        >>> reciprocal = sh3pc_inv(x)
    """
    return (x * 0 + 1) / x
