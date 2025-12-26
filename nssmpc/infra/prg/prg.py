#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch
import torchcsprng

from nssmpc.config import DEVICE, data_type, DTYPE
from nssmpc.infra.tensor import RingTensor


class PRG(object):
    """
    Seed parallel pseudo-random number generators used to generate pseudo-random numbers.

    This can be implemented using libraries such as random, PyTorch, MT, TMT, etc.
    """

    def __init__(self, kernel='AES', device=None):
        """
        Initializes a pseudo-random number generator (The default is AES).

        Creates an **AES_PRG** instance for the actual random number generation.

        Args:
            kernel (str): Specifies the random number generation algorithm to use (default is 'AES').
            device (str): Parameter of apparatus (CPU or GPU).

        Attributes:
            _prg (AES_PRG): AES_PRG instance.
            dtype (Type): The data type of the seed.

        Examples:
            >>> prg = PRG(kernel='AES')
        """
        # self.kernel = kernel
        self._prg = torch.classes.csprng_aes.AES_PRG()
        self.dtype = None

    @property
    def device(self):
        """
        Get device type.

        Returns:
            str: The device of the current PRG.

        Examples:
            >>> dev = prg.device
        """
        return self._prg.device

    @property
    def parallel_num(self):
        """
        The number of seeds that can generate random numbers simultaneously.

        Returns:
            int: The number of concurrent PRGs currently.

        Examples:
            >>> num = prg.parallel_num
        """
        return self._prg.parallel_num

    def set_seeds(self, seeds):
        """
        Set the seed for PRG.

        The seed of this PRG is parallelized, which can simultaneously generate multiple pseudo-random numbers
        corresponding to multiple seeds.

        Args:
            seeds (RingTensor or torch.Tensor): Each element inside the tensor represents a seed.

        Examples:
            >>> prg.set_seeds(torch.tensor([123, 456]))
        """
        if isinstance(seeds, RingTensor):
            self.dtype = seeds.dtype
            self._prg.set_seeds(seeds.tensor.contiguous())
        elif isinstance(seeds, torch.Tensor):
            self._prg.set_seeds(seeds.contiguous())

    def bit_random_tensor(self, bits, device=None):
        """
        Generate a tensor containing n-bit random numbers that can be generated in parallel.

        The number of parallel operations matches the number of seed values.

        Args:
            bits (int): The bit number of random numbers.
            device (str): Parameter of apparatus (CPU or GPU).

        Returns:
            torch.Tensor: A tensor containing n-bit random numbers.

        Raises:
            ValueError: If seeds are not set.

        Examples:
            >>> rand_bits = prg.bit_random_tensor(32)
        """
        if self.parallel_num == 0:
            raise ValueError("seeds is None, please set seeds first!")
        gen = self._prg.bit_random(bits)
        if device is not None and device != self.device:
            gen = gen.to(device)
        return gen

    def random_tensor(self, length, device=None):
        """
        Generate a tensor containing random numbers that can be generated in parallel.

        The number of parallel operations matches the number of seed values.

        Args:
            length (int): The length of random numbers.
            device (str): Parameter of apparatus (CPU or GPU).

        Returns:
            torch.Tensor: A tensor containing random numbers.

        Raises:
            ValueError: If seeds are not set.

        Examples:
            >>> rand_tensor = prg.random_tensor(100)
        """
        if self.parallel_num == 0:
            raise ValueError("seeds is None, please set seeds first!")
        gen = self._prg.random(length)
        if device is not None and device != self.device:
            gen = gen.to(device)
        return gen

    def bit_random(self, bits, dtype=None, device=None):
        """
        Create a RingTensor containing n-bit random numbers that can be generated in parallel.

        The number of parallel operations matches the number of seed values.

        Args:
            bits (int): The bit number of random numbers.
            dtype (Type): The data type of the seed.
            device (str): Parameter of apparatus (CPU or GPU).

        Returns:
            RingTensor: A RingTensor containing random numbers with n bits.

        Examples:
            >>> ring_rand = prg.bit_random(32)
        """
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        gen = self.bit_random_tensor(bits)
        return RingTensor(gen, dtype, device)

    def random(self, length, dtype=None, device=None):
        """
        Create a RingTensor containing random numbers that can be generated in parallel.

        The length is determined by the ``length`` parameter.

        Args:
            length (int): The length of the random numbers.
            dtype (Type): The data type of the seed.
            device (str): Parameter of apparatus (CPU or GPU).

        Returns:
            RingTensor: A RingTensor containing random numbers with the specified shape.

        Examples:
            >>> ring_rand = prg.random(100)
        """
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        gen = self.random_tensor(length)
        return RingTensor(gen, dtype, device)


class MT19937_PRG():
    """
    The Mersenne Twister algorithm is used to generate pseudo-random numbers.
    """

    def __init__(self, dtype=data_type, device=DEVICE):
        """
        Initialize the MT19937 PRG.

        Args:
            dtype (Type): The data type of the seed.
            device (str): Parameter of apparatus (CPU or GPU).

        Examples:
            >>> mt_prg = MT19937_PRG()
        """
        self.dtype = dtype
        self.device = device
        self.generator = None

    def set_seeds(self, seed):
        """
        Create a Mersenne Twister random number generator.

        Args:
            seed (int): A seed used to initialize a random number generator.

        Examples:
            >>> mt_prg.set_seeds(12345)
        """
        self.generator = torchcsprng.create_mt19937_generator(seed)

    def random(self, length, dtype=DTYPE):
        """
        Create a RingTensor containing random numbers.

        First, create an empty tensor of the specified length, then use ``self.generator`` to generate random numbers.

        Args:
            length (int): Specifies the length of the generated random number.
            dtype (str): Specifies the type of data to return.

        Returns:
            RingTensor: A RingTensor containing random numbers with the specified shape.

        Examples:
            >>> rand_nums = mt_prg.random(10)
        """
        return RingTensor(
            torch.empty(length, dtype=self.dtype, device=self.generator.device).random_(torch.iinfo(self.dtype).min,
                                                                                        to=None,
                                                                                        generator=self.generator),
            dtype)
