#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
Used to generate key pairs in the B2A operation.
"""
from NssMPC.common.ring.ring_tensor import RingTensor

from NssMPC.crypto.aux_parameter import Parameter


class B2AKey(Parameter):
    """
    This is a parameter class for generating keys, using a static method to generate key pairs, and arithmetic secret sharing.
    """

    def __init__(self):
        """
        Attribute:
            *  **r** (*ArithmeticSecretSharing*): Generated random numbers, used to obfuscate real input data in the secure B2A protocol.
        """
        self.r = None

    @staticmethod
    def gen(num_of_params):
        """
        Generate a pair of B2AKey key objects, and use secret sharing technology to split the random number r into two parts

        First, use the :meth:`~NssMPC.common.ring.ring_tensor.RingTensor.random` method to generate an array of
        random integers in the range [0, 2] with the length ``num_of_params``. After creating two B2AKey instances,
        ``k0`` and ``k1``, the random number is shared in two parts using share and assigned to ``k0.r`` and ``k1.r``, respectively

        :param num_of_params: The number of random numbers to be generated
        :type num_of_params: int
        :return: Shared key pairs k0 and k1
        :rtype: tuple
        """
        r = RingTensor.random([num_of_params], down_bound=0, upper_bound=2)
        k0, k1 = B2AKey(), B2AKey()
        from NssMPC.crypto.primitives.arithmetic_secret_sharing import ArithmeticSecretSharing
        k0.r, k1.r = ArithmeticSecretSharing.share(r, 2)

        return k0, k1
