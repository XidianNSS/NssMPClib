import math

import torch

from config.base_configs import data_type, RING_MAX


def cuda_matmul(x, y):
    if data_type is torch.int32:
        return cuda_matmul_32(x, y)
    else:  # data_type is torch.int64
        return cuda_matmul_64(x, y)


def cuda_matmul_32(x, y):
    """
    Perform multiplication of two int32 integer matrices on CUDA.

    Since CUDA does not directly support the multiplication of integer matrices, the precision is limited
    by considering the direct conversion to 64-bit floating-point numbers for arithmetic operations.
    So for matrices of type int32,
    the function first divides each element of the input matrix into high (8bit) and low (24bit) parts.
    Then, four floating-point matrix multiplications are performed using these high and low bit elements.
    Finally, these four results are added together to get the final integer matrix multiplication result.

    Args:
       x (torch.Tensor): the first integer matrix
       y (torch.Tensor): the second integer matrix

    Returns:
       torch.Tensor: the result of x@y, which is an integer matrix.
    """
    tag = 2 ** 24

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
    Since CUDA does not directly support the multiplication of integer matrices, the precision is limited
    by considering the direct conversion to 64-bit floating-point numbers for arithmetic operations.
    So for matrices of type int64,
    split each matrix into 4 blocks of 16 bits each, operate separately and then sum to get the final result.
    Refer to CrypTen: Secure Multi-Party Computation Meets Machine Learning
    https://arxiv.org/abs/2109.00984

    Args:
        x (torch.Tensor): the first integer matrix
        y (torch.torch.Tensor): the second integer matrix

    Returns:
        torch.Tensor: the result of x@y, which is an integer matrix.
    """
    block_num = 4

    # 拆分成4份
    x_block = split_matrix(x, block_num)
    y_block = split_matrix(y, block_num)

    result = 0
    for i in range(block_num):
        for j in range(block_num):
            if (i + j) * block_num >= 64:  # BIT_LEN == 64
                continue
            shift = (i + j) * 16  # BIT_LEN / block_num
            result += torch.matmul(x_block[i], y_block[j]).long() << shift

    return result


def split_matrix(x, block_num, bit_len=64):
    """
    Split the matrix by tag

    Args:
        x: the matrix to split
        block_num: the number of blocks to split the matrix into
        bit_len: the original bit length
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
