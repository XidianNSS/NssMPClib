#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch
from NssMPC import RingTensor
from NssMPC.config import SCALE_BIT, data_type
from NssMPC.crypto.aux_parameter import Parameter
from NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key import LookUpKey
from NssMPC.crypto.primitives.function_secret_sharing.dicf import SigmaDICFKey


class ReciprocalSqrtKey(Parameter):
    """
    A key management structure for dealing with square root reciprocal and negative index values.
    """

    def __init__(self):
        """
        ATTRIBUTES:
            * **neg_exp2_look_up_key** (:class:`~NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key.LookUpKey`): Store the search key for the negative exponent values
            * **rec_sqrt_look_up_key** (:class:`~NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key.LookUpKey`): Store the search key for the reciprocal square root
            * **sigma_key** (:class:`~NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dicf_key.SigmaDICFKey`): The Sigma protocol generates the key of DICF.
            * **neg_exp2_table** (:class:`~NssMPC.common.ring.ring_tensor.RingTensor`): A lookup table for storing the negative exponent values.
            * **rec_sqrt_table** (:class:`~NssMPC.common.ring.ring_tensor.RingTensor`): A lookup table for storing the reciprocal of square roots.
        """
        self.neg_exp2_look_up_key = LookUpKey()
        self.rec_sqrt_look_up_key = LookUpKey()
        self.sigma_key = SigmaDICFKey()
        self.neg_exp2_table = None
        self.rec_sqrt_table = None

    def __getitem__(self, item):
        """
        This method overrides ``__getitem__`` to allow an instance of ReciprocalSqrtKey to be accessed via an index.
        It calls the ``__getitem__`` method of the parent class and assigns ``neg_exp2_table`` and ``rec_sqrt_table`` to the
        returned key. This allows each key to access the same lookup table.

        :param item: Index to find
        :type item: int
        :return: Key corresponding to the index
        :rtype: ReciprocalSqrtKey
        """
        key = super(ReciprocalSqrtKey, self).__getitem__(item)
        key.neg_exp2_table = self.neg_exp2_table
        key.rec_sqrt_table = self.rec_sqrt_table
        return key

    def __len__(self):
        """
        :return: The length of sigma_key, which means that sigma_key should be an object whose length can be calculated.
        :rtype: int
        """
        return len(self.sigma_key)

    @staticmethod
    def gen(num_of_keys, in_scale_bit=SCALE_BIT, out_scale_bit=SCALE_BIT):
        """
        Used to generate two ReciprocalSqrtKey instances and initialize their attributes.

        Call the function  :func:`_create_neg_exp2_table` and :func:`_create_rsqrt_table` to generate the negative
        exponential lookup table and the reciprocal square root lookup table. After creating two ReciprocalSqrtKey
        instances k0 and k1, a lookup key with negative exponents and reciprocal square roots is generated for each
        instance, as well as a SIGMA key.

        :param num_of_keys: the number of keys
        :type num_of_keys: int
        :param in_scale_bit: Scale of input bits
        :type in_scale_bit: int
        :param out_scale_bit: Scale of input and output bits
        :type out_scale_bit: int
        :return: ReciprocalSqrtKey key pair
        :rtype: tuple
        """
        down_bound = -SCALE_BIT
        upper_bound = SCALE_BIT + 1

        neg_exp2_table = _create_neg_exp2_table(down_bound, upper_bound)
        rec_sqrt_table = _create_rsqrt_table(in_scale_bit, out_scale_bit)
        k0, k1 = ReciprocalSqrtKey(), ReciprocalSqrtKey()
        k0.neg_exp2_table = k1.neg_exp2_table = neg_exp2_table
        k0.rec_sqrt_table = k1.rec_sqrt_table = rec_sqrt_table

        k0.neg_exp2_look_up_key, k1.neg_exp2_look_up_key = LookUpKey.gen(num_of_keys, down_bound, upper_bound)
        k0.rec_sqrt_look_up_key, k1.rec_sqrt_look_up_key = LookUpKey.gen(num_of_keys, 0, 8191)
        k0.sigma_key, k1.sigma_key = SigmaDICFKey.gen(num_of_keys)

        return k0, k1


def _create_rsqrt_table(in_scale_bit=SCALE_BIT, out_scale_bit=SCALE_BIT):
    """
    This function is used to create a reciprocal square root lookup table.

    Generates a square root reciprocal table based on the calculated ``q`` and converts it to the specified data type(RingTensor).

    :param in_scale_bit: The scale of the input
    :type in_scale_bit: int
    :param out_scale_bit: The scale of the output
    :type out_scale_bit: int
    :return: Square root reciprocal table
    :rtype: RingTensor
    """
    i = torch.arange(0, 2 ** 13 - 1, dtype=torch.float64)
    e = i % 64
    m = i // 64
    q = 2 ** e * (1 + m / 128)

    rec_sqrt_table = torch.sqrt(2 ** in_scale_bit / q) * 2 ** out_scale_bit
    rec_sqrt_table = rec_sqrt_table.to(data_type)
    rec_sqrt_table = RingTensor(rec_sqrt_table, 'float')

    return rec_sqrt_table


def _create_neg_exp2_table(down_bound, upper_bound):
    """
    This function is used to create a negative exponential lookup table.

    First, creates a tensor that ranges from ``down_bound`` to ``upper_bound``.  Then use :meth:`~NssMPC.common.ring.ring_tensor.RingTensor.exp2` to compute and generate a lookup table with negative index values

    :param down_bound: Lower bound
    :type down_bound: int
    :param upper_bound: Upper bound
    :type upper_bound: int
    :return: Negative index lookup table
    :rtype: RingTensor
    """
    i = RingTensor.arange(down_bound, upper_bound)
    table = RingTensor.exp2(-i)
    return table
