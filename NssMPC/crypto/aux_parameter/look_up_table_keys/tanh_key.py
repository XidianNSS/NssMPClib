#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch
from NssMPC import RingTensor
from NssMPC.config import data_type, DEVICE, HALF_RING, BIT_LEN, SCALE_BIT, TANH_TABLE_BIT
from NssMPC.crypto.aux_parameter import Parameter, SigmaDICFKey, DPFKey
from NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key import LookUpKey
from NssMPC.crypto.aux_parameter.select_keys import SelectLinKey


# TODO: 换一种查表法看哪个更快
class TanhKey(Parameter):
    """
    The class is a structure designed to handle keys and lookup tables associated with Tanh activation functions.
    """

    def __init__(self):
        """
        ATTRIBUTES:
            * **look_up_key** (:class:`~NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key.LookUpKey`): The key of look up table.
            * **sigma_key** (:class:`~NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dicf_key.SigmaDICFKey`): The Sigma protocol generates the key of DICF.
            * **select_lin_key** (:class:`~NssMPC.crypto.aux_parameter.select_keys.selectlin_key.SelectLinKey`): Linearly select the key of the protocol.
            * **look_up_table** (:class:`~NssMPC.common.ring.ring_tensor.RingTensor`): A table that holds precomputed values for the GeLU function.
        """
        self.look_up_key = LookUpKey()
        self.sigma_key = SigmaDICFKey()
        self.select_lin_key = SelectLinKey()
        self.look_up_table = None

    def __len__(self):
        """
        :return: The length of sigma_key, which means that sigma_key should be an object whose length can be calculated.
        :rtype: int
        """
        return len(self.sigma_key)

    def __getitem__(self, item):
        """
        Get a specific key through the index and assign look_up_table to it, ensuring that each key has access to the lookup table.

        :param item: Index to find
        :type item: int
        :return: Key corresponding to the index
        :rtype: TanhKey
        """
        key = super(TanhKey, self).__getitem__(item)
        key.look_up_table = self.look_up_table
        return key

    @staticmethod
    def _gen_sigma_cmp_key(num_of_keys, table_scale_bit):
        """
        Generate the key associated with the SigmaDICFKey.

        First, the sigma keys ``k0`` and ``k1`` are initialized, a random tensor within a specified range is generated using
        the :meth:`~NssMPC.common.ring.ring_tensor.RingTensor.random` method, and ``k0`` and ``k1`` are shared as function secrets. Then, the range of ``input_r_in`` is
        reduced by right-shift operation, and the DPF key is generated by :meth:`~NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dpf_key.DPFKey.gen` after comparison point ``y1`` is
        generated. Using :meth:`~NssMPC.common.ring.ring_tensor.RingTensor.signbit` to compute the input signbit, and generate a random binary tensor ``c0``, based on the
        signbit ``c`` and ``c0`` XOR to generate ``c1``, ``c0`` and ``c1`` are assigned to ``k0.c`` and ``k1.c`` respectively.

        :param num_of_keys: The number of generated keys
        :type num_of_keys: int
        :param table_scale_bit: The number of bits used for scaling, which affects the range of small random inputs
        :type table_scale_bit: int
        :return: SigmaDICFKey key pair, reduced confusion factor.
        :rtype: tuple
        """
        k0, k1 = SigmaDICFKey(), SigmaDICFKey()

        input_r_in = RingTensor.random([num_of_keys], down_bound=-HALF_RING // 2, upper_bound=HALF_RING // 2)
        from NssMPC import ArithmeticSecretSharing
        k0.r_in, k1.r_in = ArithmeticSecretSharing.share(input_r_in, 2)
        k0.r_in, k1.r_in = k0.r_in.item, k1.r_in.item
        small_r = input_r_in // 2 ** (SCALE_BIT - table_scale_bit)
        small_r.bit_len = BIT_LEN - SCALE_BIT + table_scale_bit

        y1 = small_r % (HALF_RING - 1)
        k0.dpf_key, k1.dpf_key = DPFKey.gen(num_of_keys, y1, RingTensor(1))
        c = input_r_in.signbit()
        c0 = RingTensor.random([num_of_keys], down_bound=0, upper_bound=2)
        c1 = c ^ c0

        k0.c = c0
        k1.c = c1

        return k0, k1, small_r

    @staticmethod
    def gen(num_of_keys):
        """
        Generate two GeLUKey instances k0 and k1, and initialize various associated keys and lookup tables for them.

        First initialize the scale and size of the lookup table, create two GeLUKey objects ``k0`` and ``k1``,
        use the method :meth:`~NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key.LookUpKey.gen` to generate the
        lookup key, and use the function :func:`_create_tanh_table` to generate the lookup table. Generate sigma
        comparison keys using :meth:`_gen_sigma_cmp_key`. q . The q is the search range. Call the
        :meth:`~NssMPC.crypto.aux_parameter.select_keys.selectlin_key.SelectLinKey.gen` method, passing num_of_keys,
        p, and q, to generate the selected linear key. Finally, a select key pair is generated.

        .. note::
            The default scale of elements in the gelu table is 2 ** 6.

        :param num_of_keys: the number of keys
        :type num_of_keys: int
        :returns: GeLUKey key pair
        :rtype: tuple
        """
        table_scale_bit = TANH_TABLE_BIT
        table_scale = 2 ** table_scale_bit
        table_size = 2 * table_scale
        k0 = TanhKey()
        k1 = TanhKey()

        k0.look_up_key, k1.look_up_key = LookUpKey.gen(num_of_keys, 0, table_size)
        k0.look_up_table = k1.look_up_table = _create_tanh_table(table_scale_bit)

        k0.sigma_key, k1.sigma_key, select_lin_r = TanhKey._gen_sigma_cmp_key(num_of_keys, table_scale_bit)

        p = RingTensor([0, 0, -1, 1]).repeat(num_of_keys, 1)

        q = RingTensor([2 ** (table_scale_bit + 1) - 1, 2 ** (table_scale_bit + 1) - 1]).repeat(num_of_keys, 1)
        q = RingTensor.cat((q, select_lin_r.view(-1, 1), (-select_lin_r).view(-1, 1)), dim=1)

        k0.select_lin_key, k1.select_lin_key = SelectLinKey.gen(num_of_keys, p, q)
        return k0, k1


def _create_tanh_table(table_scale_bit=TANH_TABLE_BIT):
    """
    The function is used to generate a lookup table that stores the result of tanh operation.

    Use ``torch.arange`` to generate a one-dimensional tensor from 0 to 4 * table_scale. These values are then
    normalized to the range [0, 4) by dividing by table_scale. This code ensures that the input keys we create have
    the appropriate precision. After using ``torch.tanh`` to calculate the Tanh value, Convert the calculated difference to the RingTensor type.

    .. note::
        The precision of this input is determined by table_scale_bit(f), that is, the input range [0, 2 ** (f + 2))

    :param table_scale_bit: Determines the precision and size of the lookup table. (The default value is **GELU_TABLE_BIT**)
    :type table_scale_bit: int
    :returns: the table of the tanh operation
    :rtype: RingTensor
    """

    table_scale = 2 ** table_scale_bit
    table_key = torch.arange(0, 2 * table_scale, dtype=data_type, device=DEVICE) / table_scale

    table_norm_value = torch.tanh(table_key)
    table = RingTensor.convert_to_ring(table_norm_value)

    return table
