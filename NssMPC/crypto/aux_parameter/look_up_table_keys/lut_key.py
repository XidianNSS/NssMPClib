#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC import RingTensor
from NssMPC.config.runtime import ParameterRegistry
from NssMPC.crypto.aux_parameter import Parameter


@ParameterRegistry.ignore()
class LookUpKey(Parameter):
    """
    Used to generate lookup keys.
    """

    def __init__(self):
        """
        ATTRIBUTES:
            * **onehot_value** (*ArithmeticSecretSharing*): Stores uniquely thermally encoded values, converting numeric values to sparse vector form.
            * **down_bound** (*int*): Lower bound, used for range limiting.
            * **upper_bound** (*int*): Upper bound, used for range limiting.
            * **phi** (*ArithmeticSecretSharing*): A randomly generated tensor
        """
        self.onehot_value = None
        self.down_bound = None
        self.upper_bound = None
        self.phi = None

    @staticmethod
    def gen(num_of_keys, down, upper):
        """
        Create two LookUpKey instances and initialize their attributes.

        First, the upper and lower bounds are adjusted so that subsequent random number generation is only in the
        range [0, upper), and then the random tensor ``phi`` is generated to convert ``phi`` to a unique thermal
        encoding format Create two LookUpKey instances ``k0`` and ``k1``, share ``onehot_value`` and ``phi`` to
        ``k0`` and ``k1``, and set the upper and lower bounds of the two keys to the same value.

        :param num_of_keys: The number of keys generated
        :type num_of_keys: int
        :param down: Lower bound
        :type down: int
        :param upper: Upper bound
        :type upper: int
        :return: LookUpKey key pair
        :rtype: tuple
        """
        upper = upper - down
        down = 0

        phi = RingTensor.random([num_of_keys], down_bound=0, upper_bound=upper)

        k0 = LookUpKey()
        k1 = LookUpKey()

        onehot_value = RingTensor.onehot(phi, num_classes=upper)
        from NssMPC.crypto.primitives.arithmetic_secret_sharing import ArithmeticSecretSharing

        k0.onehot_value, k1.onehot_value = ArithmeticSecretSharing.share(onehot_value, 2)

        k0.phi, k1.phi = ArithmeticSecretSharing.share(phi, 2)
        k0.down_bound = k1.down_bound = down
        k0.upper_bound = k1.upper_bound = upper

        return k0, k1
