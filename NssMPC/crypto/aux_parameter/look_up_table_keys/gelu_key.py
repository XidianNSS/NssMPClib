#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch
from NssMPC import RingTensor
from NssMPC.config import data_type, DEVICE, SCALE_BIT, BIT_LEN, HALF_RING, GELU_TABLE_BIT
from NssMPC.config.runtime import ParameterRegistry
from NssMPC.crypto.aux_parameter import SigmaDICFKey, DPFKey, Parameter
from NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key import LookUpKey
from NssMPC.crypto.aux_parameter.select_keys import SelectLinKey


class GeLUKey(Parameter):
    """
    Generate keys for the Gaussian Error Linear Unit (GeLU) activation function.
    """

    @ParameterRegistry.ignore()
    class _SelectKey(Parameter):
        def __init__(self):
            self.w = None
            self.z = None

    def __init__(self):
        """
        ATTRIBUTES:
            * **look_up_key** (:class:`~NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key.LookUpKey`): The key of look up table.
            * **sigma_key** (:class:`~NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dicf_key.SigmaDICFKey`): The Sigma protocol generates the key of DICF.
            * **select_lin_key** (:class:`~NssMPC.crypto.aux_parameter.select_keys.selectlin_key.SelectLinKey`): An instance of SelectLinKey, possibly for selection operations in the computation.
            * **select_key** (:class`~NssMPC.crypto.aux_parameter.look_up_table_keys.gelu_key.GeLUKey._SelectKey`): Contains two attributes: w (weight) and z (the result of GELU operation).
            * **look_up_table** (:class:`~NssMPC.common.ring.ring_tensor.RingTensor`): A table that holds precomputed values for the GeLU function.
        """
        self.look_up_key = LookUpKey()
        self.sigma_key = SigmaDICFKey()
        self.select_lin_key = SelectLinKey()
        self.select_key = GeLUKey._SelectKey()
        self.look_up_table = None

    def __len__(self):
        """
        :return: The length of sigma_key, which means that sigma_key should be an object whose length can be calculated.
        :rtype: int
        """
        return len(self.sigma_key)

    @staticmethod
    def _gen_select_key(s_r_in, x_r_in):
        """
        Generates two selection keys based on input values ``s_r_in`` and ``x_r_in``. It uses secret sharing to split these values into two shares for secure computations.

        :param s_r_in: Blinding factor for select bit s
        :type s_r_in: RingTensor
        :param x_r_in: Blinding factor for data x
        :type x_r_in: RingTensor
        :return: shared key pairs k0 and k1
        :rtype: tuple
        """
        w = s_r_in * x_r_in
        z = 2 * s_r_in * x_r_in

        k0, k1 = GeLUKey._SelectKey(), GeLUKey._SelectKey()
        from NssMPC import ArithmeticSecretSharing
        k0.w, k1.w = ArithmeticSecretSharing.share(w, 2)
        k0.z, k1.z = ArithmeticSecretSharing.share(z, 2)

        k0.w, k1.w, k0.z, k1.z = k0.w.item, k1.w.item, k0.z.item, k1.z.item
        return k0, k1

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
        :return: SigmaDICFKey key pair, randomly generated confusion factor input_r_in, reduced confusion factor.
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

        return k0, k1, input_r_in, small_r

    @staticmethod
    def gen(num_of_keys):
        """
        Generate two GeLUKey instances k0 and k1, and initialize various associated keys and lookup tables for them.

        First initialize the scale and size of the lookup table, create two GeLUKey objects ``k0`` and ``k1``,
        use the method :meth:`~NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key.LookUpKey.gen` to generate the lookup key, and use the function :func:`_create_gelu_table` to generate the lookup
        table. Generate sigma comparison keys using :meth:`_gen_sigma_cmp_key`. q . The q is the search range. Call the
        :meth:`~NssMPC.crypto.aux_parameter.select_keys.selectlin_key.SelectLinKey.gen` method, passing num_of_keys, p, and q, to generate the selected linear key. Finally,
        a select key pair is generated.

        .. note::
            The default scale of elements in the gelu table is 2 ** 6.

        :param num_of_keys: the number of keys
        :type num_of_keys: int
        :returns: GeLUKey key pair
        :rtype: tuple

        """
        table_scale_bit = GELU_TABLE_BIT
        table_scale = 2 ** table_scale_bit
        table_size = 4 * table_scale
        k0 = GeLUKey()
        k1 = GeLUKey()

        k0.look_up_key, k1.look_up_key = LookUpKey.gen(num_of_keys, 0, table_size)
        k0.look_up_table = k1.look_up_table = _create_gelu_table(table_scale_bit)

        k0.sigma_key, k1.sigma_key, input_r_in, select_lin_r = GeLUKey._gen_sigma_cmp_key(num_of_keys, table_scale_bit)

        p = RingTensor([0, 0, -1, 1]).repeat(num_of_keys, 1)

        q = RingTensor([2 ** (table_scale_bit + 2) - 1, 2 ** (table_scale_bit + 2) - 1]).repeat(num_of_keys, 1)
        q = RingTensor.cat((q, select_lin_r.view(-1, 1), (-select_lin_r).view(-1, 1)), dim=1)

        k0.select_lin_key, k1.select_lin_key = SelectLinKey.gen(num_of_keys, p, q)

        r_in_1 = (k0.select_lin_key.d + k1.select_lin_key.d) % 2

        k0.select_key, k1.select_key = GeLUKey._gen_select_key(r_in_1, input_r_in)

        return k0, k1

    def __getitem__(self, item):
        """
        This method overrides the index operation so that the elements of GeLUKey can be accessed through the index.

        Set the look_up_table of the key to the look_up_table of the current instance to ensure that the correct lookup table is used during access.

        :param item: Index to find
        :type item: int
        :return: Key corresponding to the index
        :rtype: GeLUKey
        """
        key = super(GeLUKey, self).__getitem__(item)
        key.look_up_table = self.look_up_table
        return key


def _create_gelu_table(table_scale_bit=GELU_TABLE_BIT):
    """
    The function is used to generate a lookup table that stores the difference between the ReLU and GeLU functions.

    Use ``torch.arange`` to generate a one-dimensional tensor from 0 to 4 * table_scale. These values are then
    normalized to the range [0, 4) by dividing by table_scale. This code ensures that the input keys we create have
    the appropriate precision. After calculating the difference between ReLU and GeLU, Convert the calculated difference to the RingTensor type.

    .. note::
        * In the input domain range, ReLU(x)-GeLU(x) is non-zero on (-4, 4), and is an even function, so the input range can be determined to be [0, 4)

        * The precision of this input is determined by table_scale_bit(f), that is, the input range [0, 2 ** (f + 2))

    :param table_scale_bit: Determines the precision and size of the lookup table. (The default value is **GELU_TABLE_BIT**)
    :type table_scale_bit: int
    :returns: the table of the ReLU(x)-GeLU(x)
    :rtype: RingTensor
    """

    table_scale = 2 ** table_scale_bit
    table_key = torch.arange(0, 4 * table_scale, dtype=data_type, device=DEVICE) / table_scale
    relu = torch.nn.ReLU()
    gelu = torch.nn.GELU()

    table_norm_value = relu(table_key) - gelu(table_key)
    table = RingTensor.convert_to_ring(table_norm_value)
    return table