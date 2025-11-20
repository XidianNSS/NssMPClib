#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch

from NssMPC.config import DEVICE


# a Mersenne transformation
def _torch_int32(x):
    """
    The input value is returned without any processing.

    Args:
        x (int or torch.Tensor): Input number.

    Returns:
        int or torch.Tensor: Original value.

    Examples:
        >>> val = _torch_int32(10)
    """
    # lsb = 0xFFFFFFFF & x
    # return lsb
    return x


def _int32(x):
    """
    Limit the input value to 32 bits, ensuring that the returned value is in the range of 32-bit integers.

    Args:
        x (int): Input number.

    Returns:
        int: The 32 least significant bits of ``x``.

    Examples:
        >>> val = _int32(0x1FFFFFFFF)
    """
    # Get the 32 least significant bits.
    return int(0xFFFFFFFF & x)


class TorchMT19937:
    """
    The Mersenne Twister algorithm implementation using internal state arrays to maintain state and provide random number generation.
    """

    def __init__(self, seeds):
        """
        Initialize TorchMT19937 object.

        Initializes the state array size to 624, with the first element filled with seeds and the last 623 elements initialized to zero.
        Fills the rest of the state array with seeds according to the Mersenne Twister algorithm.

        Args:
            seeds (torch.Tensor): Seed value list (one-dimensional).

        Raises:
            ValueError: If seeds are not one-dimensional.

        Examples:
            >>> mt = TorchMT19937(torch.tensor([123]))
        """
        if len(seeds.shape) != 1:
            raise ValueError('seeds can only be one-dimensional arrays', )
        self.index = 624
        l = seeds.shape[0]
        s = seeds.unsqueeze(1)
        ex = torch.zeros([l, 623], dtype=torch.int32, device=DEVICE)
        self.mt = torch.cat([s, ex], dim=1)
        for i in range(1, 624):
            self.mt[:, i] = _torch_int32(1812433253 * (self.mt[:, i - 1] ^ self.mt[:, i - 1] >> 30) + i)

    def twist(self):
        """
        Update the state.

        Each state value is combined with the next state value and a series of bit operations are applied to generate the new value.
        A constant ``0x9908b0df`` is used to handle odd bits, introducing randomness.

        Examples:
            >>> mt.twist()
        """
        for i in range(624):
            y = _torch_int32((self.mt[:, i] & 0x80000000) +
                             (self.mt[:, (i + 1) % 624] & 0x7fffffff))
            self.mt[:, i] = self.mt[:, (i + 397) % 624] ^ y >> 1

            cw = ((y % 2 != 0) + 0) * 0x9908b0df

            self.mt[:, i] = self.mt[:, i] ^ cw

        self.index = 0
        return

    def random(self, num):
        """
        Generate random numbers.

        First checks if status update is needed (twist is called when ``self.index`` exceeds 624).
        Then, extracts ``num`` random numbers from the state array and performs a series of bit operations on them.

        Args:
            num (int): The number of random numbers.

        Returns:
            torch.Tensor: The generated random numbers.

        Examples:
            >>> rand_nums = mt.random(10)
        """
        if self.index + num - 1 >= 624:
            self.twist()
        y = self.mt[:, self.index: self.index + num]
        # y = self.mt[:, self.index]

        # Right shift by 11 bits
        y = y ^ y >> 11
        # Shift y left by 7 and take the bitwise and of 2636928640
        y = y ^ y << 7 & 2636928640
        # Shift y left by 15 and take the bitwise and of y and 4022730752
        y = y ^ y << 15 & 4022730752
        # Right shift by 18 bits
        y = y ^ y >> 18
        self.index = self.index + num
        return _torch_int32(y)

    def extract_number(self):
        """
        Single number extraction.

        Similar to the random method, but only one random number is extracted at a time.

        Returns:
            torch.Tensor: The generated random number.

        Examples:
            >>> num = mt.extract_number()
        """

        if self.index >= 624:
            self.twist()
        y = self.mt[:, self.index]
        # Right shift by 11 bits
        y = y ^ y >> 11
        # Shift y left by 7 and take the bitwise and of 2636928640
        y = y ^ y << 7 & 2636928640
        # Shift y left by 15 and take the bitwise and of y and 4022730752
        y = y ^ y << 15 & 4022730752
        # Right shift by 18 bits
        y = y ^ y >> 18
        self.index = self.index + 1
        return _torch_int32(y)
