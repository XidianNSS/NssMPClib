#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.config import SCALE_BIT
from NssMPC.config.runtime import ParameterRegistry
from NssMPC.crypto.aux_parameter import Parameter
from NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key import LookUpKey
from NssMPC.crypto.primitives.function_secret_sharing.dicf import SigmaDICFKey


class DivKey(Parameter):
    """
    The DivKey class for generating and managing division keys
    """

    @ParameterRegistry.ignore()
    class _NegExp2Key(Parameter):
        """
        Used to store keys associated with negative exponents.
        """

        def __init__(self):
            """
            ATTRIBUTES:
                * **look_up_table** (*LookUpKey*): The key that stores the lookup table
                * **table** (*RingTensor*): The lookup table
            """
            self.look_up_key = LookUpKey()
            self.table = None

        def __getitem__(self, item):
            """
            Override index access to return the key and assign the table attribute to it.

            This allows instances using NegExp2Key to access elements like a dictionary and automatically associate corresponding lookup tables.

            :param item: Index to find
            :type item: int
            :return: Key corresponding to the index
            :rtype: Parameter
            """
            key = super(DivKey._NegExp2Key, self).__getitem__(item)
            key.table = self.table
            return key

    def __init__(self):
        """
        Attribute:
            * **neg_exp2_key** (*_NegExp2Key*): Used to store keys associated with negative exponents.
            * **sigma_key** (*SigmaDICFKey*): The Sigma protocol generates the key of DICF
        """
        self.neg_exp2_key = DivKey._NegExp2Key()
        self.sigma_key = SigmaDICFKey()

    def __len__(self):
        """
        :return: The length of sigma_key, which means that sigma_key should be an object whose length can be calculated.
        :rtype: int
        """
        return len(self.sigma_key)

    @staticmethod
    def _gen_neg_exp2_key(num_of_keys):
        """
        Generates a negative exponential lookup table based on the given lower and upper limits.

        First set the upper and lower bounds of the lookup table, and then create two pairs of _NegExp2Key instances
        k0 and k1. Then use :meth:`~NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key.LookUpKey.gen` to create
        lookup keys associated with negative exponens and assign the results to ``k0.look_up_key`` and
        ``k1.look_up_key``, respectively. Finally, :func:`_create_neg_exp2_table` is called to generate a negative
        exponential lookup table assigned to ``k0.table`` and ``k1.table`` respectively.

        :param num_of_keys: The number of keys to be generated.
        :type num_of_keys: int
        :return: _NegExp2Key key pairs k0 and k1
        :rtype: tuple
        """
        down_bound = -SCALE_BIT
        upper_bound = SCALE_BIT + 1

        k0, k1 = DivKey._NegExp2Key(), DivKey._NegExp2Key()

        k0.look_up_key, k1.look_up_key = LookUpKey.gen(num_of_keys, down_bound, upper_bound)
        k0.table = k1.table = _create_neg_exp2_table(down_bound, upper_bound)

        return k0, k1

    @staticmethod
    def gen(num_of_keys):
        """
        Generate a pair of DivKey instances and initialize the associated negative exponential and sigma keys for each of them.

        First create two DivKey instances ``k0`` and ``k1``, then call the :meth:`_gen_neg_exp2_key` method to
        generate a pair of negative exponential lookup keys and assign the results to ``k0.neg_exp2_key`` and
        ``k1.neg_exp2_key``, respectively. The
        :meth:`~NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dicf_key.SigmaDICFKey.gen` method is then
        called to generate a SigmaDICFKey pair and assign the result to ``k0.sigma_key`` and ``k1.sigma_key``,
        respectively

        :param num_of_keys: The number of keys to be generated.
        :type num_of_keys: int
        :return: DivKey key pairs k0 and k1
        :rtype: tuple
        """
        k0, k1 = DivKey(), DivKey()
        k0.neg_exp2_key, k1.neg_exp2_key = DivKey._gen_neg_exp2_key(num_of_keys)
        k0.sigma_key, k1.sigma_key = SigmaDICFKey.gen(num_of_keys)

        return k0, k1


def _create_neg_exp2_table(down_bound, upper_bound):
    """
    This function is used to generate a negative exponential lookup table.

    First create a tensor from *down_bound* to *upper_bound*, then use the method
    :meth:`~NssMPC.common.ring.ring_tensor.RingTensor.exp2` to compute the quadratic exponent of **-i** and return
    the result.

    :param down_bound: Represents the lower bound of the lookup table
    :type down_bound: int or float
    :param upper_bound: Represents the upper bound of the lookup table
    :type upper_bound: int or float
    :return: The negative exponent of each integer from down_bound to upper_bound.
    :rtype: RingTensor
    """
    i = RingTensor.arange(down_bound, upper_bound)
    table = RingTensor.exp2(-i)
    return table
