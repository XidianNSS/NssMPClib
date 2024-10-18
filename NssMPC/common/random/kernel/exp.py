#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch

from NssMPC.config import DEVICE


def _torch_int32(x):
    """
    The input value is returned without any processing.

    :param x: Input number
    :type x: int
    :return: Original value
    """
    # lsb = 0xFFFFFFFF & x
    # return lsb
    return x


def _int32(x):
    """
    Limit the input value to 32 bits, ensuring that the returned value is in the range of 32-bit integers.

    :param x: Input number
    :return: Gets the 32 least significant bits of *x*
    """
    return x & 0xFFFFFFFF


class TMT:
    def __init__(self, seeds):
        """
        TMT is a simplified implementation of Mersenne Twister.

        ATTRIBUTES:
            * **l** (*int*): Seed length
            * **s** (*list*): Seed Value
            * **bias** (*int*): Constant deviation value
        """
        if len(seeds.shape) != 1:
            raise ValueError('seeds can only be one-dimensional arrays', )
        self.l = seeds.shape[0]
        self.s = seeds
        self.bias = 1234

    def random(self, num):
        """
        Generate the 64 bits random number

        :param num: the number of random numbers to be generated for each seed
        :type num: int
        :returns: A list of random numbers with the same length(num) as the seed number and parallelism.
        """
        out = torch.empty([self.l, num], dtype=torch.int64, device=DEVICE)
        cur = self.s + self.bias
        for i in range(num + 1):
            cur = 1812433253 * (cur ^ cur >> 30) + i
            y = cur
            # Right shift by 11 bits
            y = y ^ y >> 11
            # Shift y left by 7 and take the bitwise and of 2636928640
            y = y ^ y << 7 & 2636928640
            # Shift y left by 15 and take the bitwise and of y and 4022730752
            y = y ^ y << 15 & 4022730752
            # Right shift by 18 bits
            y = y ^ y >> 18
            if i > 0:
                out[:, i - 1] = y
            #
            # out[:, i] = y

        return out
