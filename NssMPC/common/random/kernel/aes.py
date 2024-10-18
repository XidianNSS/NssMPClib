"""
.. note::
    * csprng only support the Python3.8.0
    * AES requires two int64 (128bit) per seed
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torchcsprng as csprng


class AES:
    def __init__(self, seeds):
        """
        Using AES algorithm to generate random numbers.

        :param seeds: seeds generated from pseudo-random numbers
        :type seeds: int
        """
        self.s = seeds

    def bit_random(self, bits):
        """
        Cryptographically Secure Pseudo-Random Number Generator for PyTorch

        Each seed generates 'bits' bits of pseudo-random numbers, carried in int64, independent of *BIT_LEN*

        :param bits: The bit width of the random number is required
        :type bits: int
        :return: The first dimension is the number of parallelizations (seeds), and the second dimension is the int64 required to carry bits of random numbers
        :rtype: torch.Tensor
        """
        random_out = csprng.random_repeat(self.s, bits)
        return random_out
