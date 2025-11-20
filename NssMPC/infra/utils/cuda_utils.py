#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import math
from typing import Callable

import torch

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

from NssMPC.config import data_type, RING_MAX


def cuda_matmul(x, y):
    """
    Performs matrix multiplication using CUDA-optimized logic for integer types.

    Handles precision limitations by converting to 64-bit floats or splitting blocks
    depending on the data type.

    Args:
        x (torch.Tensor): The multiplicand matrix.
        y (torch.Tensor): The multiplier matrix.

    Returns:
        torch.Tensor: The product of the matrices.

    Examples:
        >>> result = cuda_matmul(x, y)
    """
    if data_type is torch.int32:
        return cuda_matmul_32(x, y)
    else:  # data_type is torch.int64
        return cuda_matmul_64(x, y)


def cuda_matmul_32(x, y):
    """
    Performs multiplication of two int32 integer matrices on CUDA.

    Splits elements into high and low parts to perform floating-point operations
    and reconstructs the integer result.

    Args:
        x (torch.Tensor): The first integer matrix (int32).
        y (torch.Tensor): The second integer matrix (int32).

    Returns:
        torch.Tensor: The result of x @ y as an int32 matrix.

    Examples:
        >>> result = cuda_matmul_32(x, y)
    """
    tag = 2 ** 16

    x_high = torch.floor(x / tag).to(torch.float64)
    x_low = (x - x_high * tag).to(torch.float64)

    y_high = torch.floor(y / tag).to(torch.float64)
    y_low = (y - y_high * tag).to(torch.float64)

    result = (torch.matmul(x_high, y_high) * tag * tag % RING_MAX +
              torch.matmul(x_high, y_low) * tag % RING_MAX +
              torch.matmul(x_low, y_high) * tag % RING_MAX +
              torch.matmul(x_low, y_low)) % RING_MAX

    # You need to convert to int64 and then to int32, otherwise you get an error.
    return result.to(torch.int64).to(torch.int32)


def cuda_matmul_64(x, y):
    """
    Performs multiplication of two int64 integer matrices on CUDA.

    Splits matrices into 4 blocks of 16 bits each to handle 64-bit integer multiplication
    using floating-point hardware.

    Args:
        x (torch.Tensor): The multiplicand matrix (int64).
        y (torch.Tensor): The multiplier matrix (int64).

    Returns:
        torch.Tensor: The product of the matrices.

    Examples:
        >>> result = cuda_matmul_64(x, y)
    """
    block_num = 4
    # 拆分成4份
    x_block = split_matrix(x, block_num)
    y_block = split_matrix(y, block_num)

    result = 0
    for i in range(block_num):
        for j in range(block_num):
            if (i + j) * 16 >= 64:  # BIT_LEN == 64
                continue
            shift = (i + j) * 16  # BIT_LEN / block_num
            result += torch.matmul(x_block[i], y_block[j]).long() << shift

    return result


def split_matrix(x, block_num, bit_len=64):
    """
    Splits a matrix into multiple blocks based on bit length.

    Args:
        x (torch.Tensor): The matrix to split.
        block_num (int): The number of blocks to split into.
        bit_len (int, optional): The total bit length of the elements. Defaults to 64.

    Returns:
        list[torch.Tensor]: A list of tensor blocks.

    Examples:
        >>> blocks = split_matrix(x, 4)
    """
    block_size = math.ceil(bit_len / block_num)
    tag = 2 ** block_size

    x_ = x
    x_block = []

    for _ in range(block_num - 1):
        x_block.append((x_ % tag).to(torch.float64))
        x_ = x_ >> block_size
    x_block.append(x_.to(torch.float64))
    return x_block


if _HAS_TRITON:
    def rotate_by_triton(inputs, shifts, block_n: int = 256):
        """
        Rotates rows of a tensor using a Triton kernel.

        Args:
            inputs (torch.Tensor): The input tensor to rotate.
            shifts (torch.Tensor): The shift amounts for each row.
            block_n (int, optional): The block size for the kernel. Defaults to 256.

        Returns:
            torch.Tensor: The rotated tensor.

        Examples:
            >>> out = rotate_by_triton(inputs, shifts)
        """
        n = inputs.shape[1]
        b = shifts.numel()

        s = shifts.to(torch.int32)
        s = ((s % n) + n) % n

        out = torch.empty((b, n), dtype=inputs.dtype, device=inputs.device)
        grid = (b, (n + block_n - 1) // block_n)
        _roll_kernel[grid](inputs, out, s, n, block_n)
        return out


    @triton.jit
    def _roll_kernel(x_ptr, out_ptr, shifts_ptr, n: tl.constexpr, block_n: tl.constexpr):
        row_id = tl.program_id(0)
        col_blk = tl.program_id(1)

        cols = col_blk * block_n + tl.arange(0, block_n)
        mask = cols < n

        s = tl.load(shifts_ptr + row_id)

        src = cols + (n - s)
        src = tl.where(src >= n, src - n, src)

        vals = tl.load(x_ptr + src, mask=mask, other=0)
        tl.store(out_ptr + row_id * n + cols, vals, mask=mask)


    cuda_rotate: Callable = rotate_by_triton
else:
    def rotate_by_torch(inputs, shifts):
        """
        Rotates rows of a tensor using standard PyTorch operations.

        Args:
            inputs (torch.Tensor): The input tensor to rotate.
            shifts (torch.Tensor): The shift amounts for each row.

        Returns:
            torch.Tensor: The rotated tensor.

        Examples:
            >>> out = rotate_by_torch(inputs, shifts)
        """
        n = inputs.shape[1]
        rows = torch.arange(inputs.shape[0]).view(-1, 1)
        indices = (torch.arange(n, device=shifts.device) - shifts.view(-1, 1)) % n
        result = inputs[rows, indices]
        return result


    cuda_rotate: Callable = rotate_by_torch
