import math

import torch
import torchcsprng as csprng

from config.base_configs import DEVICE, BIT_LEN, data_type

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
        random_out = self._random_repeat(bits)
        return random_out

    def _random_repeat(self, bits):
        shape = list(self.s.shape)
        if len(self.s.shape) != 128 // BIT_LEN:
            self.s = self.s.view(-1, 128 // BIT_LEN)
        # the byte of element
        element_byte = BIT_LEN // 8

        num_of_slice = self.s.shape[1]

        block_num = 128 // BIT_LEN

        if num_of_slice % block_num:
            padding_num = block_num - num_of_slice % block_num
            padding_seed = torch.zeros([self.s.shape[0], padding_num], dtype=self.s.dtype, device=DEVICE)

            self.s = torch.cat([self.s, padding_seed], dim=1)

        num_of_slice = self.s.shape[1]

        # the byte of seed:
        seed_byte = element_byte * num_of_slice
        # byte required:
        desired_byte = math.ceil(bits / 8)

        output_byte = math.ceil(desired_byte / 16) * 16
        # The number of elements to be encrypted
        output_num = output_byte // element_byte

        # How many output elements (integer multiples of 16 bytes) the current seed can encrypt and fill at a time:
        each_gen_num = (math.ceil(seed_byte / 16) * 16) // element_byte

        out_tensor = torch.empty([self.s.shape[0], output_num], dtype=data_type, device=DEVICE)

        last_out = self.s

        generated_num = 0
        key = torch.tensor([1, 1], device=DEVICE)

        # Secondary pseudo-random generation using the random numbers generated in the previous round each time
        while generated_num < output_num:
            inner_encrypt = torch.empty([self.s.shape[0], each_gen_num], dtype=data_type, device=DEVICE)
            # Encrypt the output of the previous round
            csprng.encrypt(last_out, inner_encrypt, key, "aes128", "ecb")
            if generated_num + each_gen_num > output_num:
                each_gen_num = output_num - generated_num
                out_tensor[:, generated_num: generated_num + each_gen_num] = inner_encrypt[:, :each_gen_num]
                break
            # Assign the contents of this round of calculations to the output container
            out_tensor[:, generated_num: generated_num + each_gen_num] = inner_encrypt
            last_out = inner_encrypt

            generated_num += each_gen_num

        out_tensor = out_tensor.view(shape[:-1] + [-1])

        return out_tensor
