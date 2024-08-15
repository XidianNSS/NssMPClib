import torch
import torchcsprng
from torchcsprng import PRG as AES_PRG

from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.config import DEVICE, data_type, DTYPE


class PRG(object):
    """
    Seed parallel pseudo-random number generators used to generate pseudo-random numbers,
    which can be implemented using libraries such as random, PyTorch, MT, TMT, etc.
    """

    def __init__(self, kernel='AES', device=None):
        self.kernel = kernel
        self._prg = AES_PRG()
        self.dtype = None

    @property
    def device(self):
        return self._prg.device

    @property
    def parallel_num(self):
        return self._prg.parallel_num

    def set_seeds(self, seeds):
        """
        Set the seed for PRG
        The seed of this PRG is parallelized, which can simultaneously generate multiple pseudo-random numbers
        corresponding to multiple seeds.
        Here, 'seeds' is a tensor, where each element inside the tensor represents a seed.

        Args:
            seeds: each element inside the tensor represents a seed.
        """
        if isinstance(seeds, RingTensor):
            self.dtype = seeds.dtype
            self._prg.set_seeds(seeds.tensor)
        elif isinstance(seeds, torch.Tensor):
            self._prg.set_seeds(seeds)

    def bit_random_tensor(self, bits, device=None):
        """
        Generate a tensor containing n-bit random numbers that can be generated in parallel,
        with the number of parallel operations matching the number of seed values.
        Args:
            bits: the bit number of random numbers.
            device:

        Returns:

        """
        if self.parallel_num == 0:
            raise ValueError("seeds is None, please set seeds first!")
        gen = self._prg.bit_random(bits)
        if device is not None and device != self.device:
            gen = gen.to(device)
        return gen

    def random_tensor(self, length, device=None):
        """
        Generate a tensor containing n-bit random numbers that can be generated in parallel,
        with the number of parallel operations matching the number of seed values.
        Args:
            bits: the bit number of random numbers.
            device:

        Returns:

        """
        if self.parallel_num == 0:
            raise ValueError("seeds is None, please set seeds first!")
        gen = self._prg.random(length)
        if device is not None and device != self.device:
            gen = gen.to(device)
        return gen

    def bit_random(self, bits, dtype=None, device=None):
        """
        Create a RingTensor containing n-bit random numbers that can be generated in parallel,
        with the number of parallel operations matching the number of seed values.

        Args:
            bits: the bit number of random numbers
            dtype
            device

        Returns:
            A RingTensor, which is a random number with n bits
        """
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        gen = self.bit_random_tensor(bits)
        return RingTensor(gen, dtype, device)

    def random(self, length, dtype=None, device=None):
        """
        Create a RingTensor containing random numbers that can be generated in parallel,
        with the number of parallel operations matching the number of seed values.

        Args:
            length: the length of the random numbers
            dtype
            device

        Returns:
            A RingTensor, which is a random number with the specified shape
        """
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        gen = self.random_tensor(length)
        return RingTensor(gen, dtype, device)


class MT19937_PRG():
    def __init__(self, dtype=data_type, device=DEVICE):
        self.dtype = dtype
        self.device = device
        self.generator = None

    def set_seeds(self, seed):
        self.generator = torchcsprng.create_mt19937_generator(seed)

    def random(self, length, dtype=DTYPE):
        return RingTensor(
            torch.empty(length, dtype=self.dtype, device=self.device).random_(torch.iinfo(self.dtype).min, to=None,
                                                                              generator=self.generator), dtype)
