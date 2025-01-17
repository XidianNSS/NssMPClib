#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import math

import torch

from NssMPC.config import data_type, RING_MAX


def cuda_matmul(x, y):
    """
    Since CUDA does not directly support the multiplication of integer matrices, the precision is limited
    by considering the direct conversion to 64-bit floating-point numbers for arithmetic operations.

    :param x: multiplicand
    :type x: torch.Tensor
    :param y: multiplier
    :type y: torch.Tensor
    :return: Select the product result of the executed function
    :rtype: torch.tensor
    """
    if data_type is torch.int32:
        return cuda_matmul_32(x, y)
    else:  # data_type is torch.int64
        return cuda_matmul_64(x, y)


def cuda_matmul_32(x, y):
    """
    Perform multiplication of two int32 integer matrices on CUDA.

    For matrices of type int32, the function first divides each element of the input matrix into high (8bit) and low (24bit) parts.
    Then, four floating-point matrix multiplications are performed using these high and low bit elements.
    Finally, these four results are added together to get the final integer matrix multiplication result.

    :param x: the first integer matrix
    :type x: torch.Tensor
    :param y: the second integer matrix
    :type y: torch.Tensor
    :returns: the result of x@y, which is an integer matrix.
    :rtype: torch.Tensor
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
    Perform multiplication of two int64 integer matrices on CUDA.

    So for matrices of type int64, split each matrix into 4 blocks of 16 bits each, operate separately and then sum to get the final result.
    Refer to CrypTen: *Secure Multi-Party Computation Meets Machine Learning* https://arxiv.org/abs/2109.00984

    :param x: multiplicand
    :type x: torch.Tensor
    :param y: multiplier
    :type y: torch.Tensor
    :return: Select the product result of the executed function
    :rtype: torch.tensor
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
    Split the matrix by tag.

    First calculate the size of the block you want to split, then iterate to extract the value of the current block
    using *x_ % tag*, convert it to type *torch.float64*, and shift *x_* to the right to extract the next block. Add the
    last block at the end of the loop.

    :param x: the matrix to split
    :type x: torch.Tensor
    :param block_num: the number of blocks to split the matrix into
    :type block_num: int
    :param bit_len: the original bit length
    :type bit_len: int
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
