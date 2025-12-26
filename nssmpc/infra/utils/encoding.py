#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
This document provides functions for encoding RingTensors using bitwise encoding schemes.
It includes two primary encoding methods:

1. **zero_encoding**: Encodes the bits of a RingTensor where the bits with value `0` are processed and transformed.
2. **one_encoding**: Encodes the bits of a RingTensor where the bits with value `1` are processed and transformed.

Each encoding function returns an encoded tensor along with a corresponding mask that indicates
the positions of the modified bits.

These encoding schemes are used in secure multi-party computation (MPC) protocols
to ensure the privacy and security of the shared data.
"""

from nssmpc.infra.tensor import RingTensor


def zero_encoding(x: RingTensor):
    """
    Encodes the input `x` using a zero encoding scheme.

    This function encodes each bit of the input RingTensor `x` into a new RingTensor
    where the bits corresponding to `0` are replaced with either random values or
    transformations based on the subsequent bits of `x`.

    Args:
        x: The input RingTensor to be flattened and encoded.

    Returns:
        Tuple[RingTensor, RingTensor]: A tuple containing:
            - zero_encoding_list: A tensor containing the encoded values.
            - fake_mask: A mask marking the locations of encoded values for fake (zero) bits.

    Examples:
        >>> encoded, mask = zero_encoding(x)
    """
    bit_len = x.bit_len
    x = x.flatten()
    zero_encoding_list = RingTensor.empty([x.numel(), bit_len], dtype=x.dtype, device=x.device)
    cw = RingTensor.where(x > 0, 0, -1)
    for i in range(bit_len - 1, -1, -1):
        current_bit = x.get_bit(i)
        cw = cw << 1
        cur_encoded = RingTensor.where(current_bit, RingTensor.random(x.shape),
                                       (x.bit_slice(i + 1, bit_len) << 1 | 1) ^ cw)
        zero_encoding_list[:, i] = cur_encoded
    fake_mask = RingTensor.empty([x.numel(), bit_len], dtype=x.dtype, device=x.device)
    for i in range(bit_len):
        fake_mask[:, i] = 1 - x.get_bit(i)
    return zero_encoding_list, fake_mask


def one_encoding(x: RingTensor):
    """
    Encodes the input `x` using a one encoding scheme.

    This function encodes each bit of the input RingTensor `x` into a new tensor
    where the bits corresponding to `1` are processed and encoded based on a bit-slice
    transformation or left as random values.

    Args:
        x: The input RingTensor to be flattened and encoded.

    Returns:
        Tuple[RingTensor, RingTensor]: A tuple containing:
            - one_encoding_list: A tensor containing the encoded values.
            - fake_mask: A mask marking the locations of encoded values for true (one) bits.

    Examples:
        >>> encoded, mask = one_encoding(x)
    """
    bit_len = x.bit_len
    x = x.flatten()
    one_encoding_list = RingTensor.empty([x.numel(), bit_len], dtype=x.dtype, device=x.device)
    cw = RingTensor.where(x > 0, 0, -1)
    for i in range(bit_len - 1, -1, -1):
        current_bit = x.get_bit(i)
        cur_encoded = RingTensor.where(current_bit, x.bit_slice(i, bit_len), RingTensor.random(x.shape))
        cw = cw << 1
        if i == 0:
            cw = 0
        one_encoding_list[:, i] = cur_encoded ^ cw
    fake_mask = RingTensor.empty([x.numel(), bit_len], dtype=x.dtype, device=x.device)
    for i in range(bit_len):
        fake_mask[:, i] = x.get_bit(i)
    return one_encoding_list, fake_mask
