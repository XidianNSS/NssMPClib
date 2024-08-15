import torchcsprng as csprng

"""
Using AES algorithm to generate random numbers
csprng only support the Python3.8.0
Cryptographically Secure Pseudo-Random Number Generator for PyTorch
"""


class AES:
    def __init__(self, seeds):
        """
        AES requires two int64 (128bit) per seed

        Args:
            seeds: seeds generated from pseudo-random numbers
        """
        self.s = seeds

    def bit_random(self, bits):
        """
        Each seed generates 'bits' bits of pseudo-random numbers, carried in int64, independent of BIT_LEN
        :param bits: 需要随机数的位宽
        :return: tensor，第一个维度是并行化的数量（种子个数），第二个维度是承载bits位随机数所需要的int64们
        """
        random_out = csprng.random_repeat(self.s, bits)
        return random_out
