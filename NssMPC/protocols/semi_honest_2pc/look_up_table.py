#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from typing import Tuple
from NssMPC import RingTensor
from NssMPC.config import DEBUG_LEVEL, data_type
from NssMPC.infra.mpc.param_provider.parameter import Parameter, ParameterRegistry
from NssMPC.infra.tensor import RingTensor


@ParameterRegistry.ignore()
class LookUpKey(Parameter):
    """Used to generate lookup keys.

    Attributes:
        onehot_value (ArithmeticSecretSharing): Stores uniquely thermally encoded values, converting numeric values to sparse vector form.
        down_bound (int): Lower bound, used for range limiting.
        upper_bound (int): Upper bound, used for range limiting.
        phi (ArithmeticSecretSharing): A randomly generated tensor.
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
    def gen(num_of_keys, down, upper)-> Tuple['LookUpKey', 'LookUpKey']:
        """Creates two LookUpKey instances and initializes their attributes.

        First, the upper and lower bounds are adjusted so that subsequent random number generation is only in the
        range [0, upper), and then the random tensor ``phi`` is generated to convert ``phi`` to a unique thermal
        encoding format Create two LookUpKey instances ``k0`` and ``k1``, share ``onehot_value`` and ``phi`` to
        ``k0`` and ``k1``, and set the upper and lower bounds of the two keys to the same value.

        Args:
            num_of_keys (int): The number of keys generated.
            down (int): Lower bound.
            upper (int): Upper bound.

        Returns:
            LookUpKey key pair.

        Examples:
            >>> k0, k1 = LookUpKey.gen(10, 0, 100)
        """
        upper = upper - down
        down = 0

        phi = RingTensor.random([num_of_keys], down_bound=0, upper_bound=upper)

        k0 = LookUpKey()
        k1 = LookUpKey()

        onehot_value = RingTensor.onehot(phi, num_classes=upper)
        from NssMPC.primitives.secret_sharing import AdditiveSecretSharing

        k0.onehot_value, k1.onehot_value = AdditiveSecretSharing.share(onehot_value, 2)

        k0.phi, k1.phi = AdditiveSecretSharing.share(phi, 2)
        k0.down_bound = k1.down_bound = down
        k0.upper_bound = k1.upper_bound = upper

        return k0, k1


class LookUp(object):
    """This class defines a look-up table (LUT) used for efficiently calculating values of functions such as `tanh`."""

    @staticmethod
    def eval(x, key: LookUpKey, table: RingTensor)-> RingTensor:
        """Uses the look-up table (LUT) to compute function values efficiently.

        This method finds the function value corresponding to `x` in the provided `table`,
        reducing computation overhead compared to direct function calculation.

        Args:
            x (RingTensor): The input parameter for the function.
            key: The key used in the look-up table algorithm.
            table: A `RingTensor` representing the look-up table containing precomputed function values.

        Returns:
            The result of the function evaluation, represented as a `RingTensor`.

        Examples:
            >>> res = LookUp.eval(x, key, table)
        """
        shape = x.shape
        x = x.flatten()

        key.phi *= x.scale  # TODO: 临时修正
        key.phi.dtype = x.dtype
        x_shift_shared = key.phi - x
        x_shift = x_shift_shared.restore().convert_to_real_field().to(data_type)
        y = key.onehot_value
        if DEBUG_LEVEL == 2:
            y = y.reshape(1, -1)

        u = x.__class__.rotate(y, shifts=-x_shift)  # TODO: GPU环境下，先转cpu，算完乘法后再转gpu或许更快
        res = (u * table).sum(-1)
        res.dtype = x.dtype

        return res.reshape(shape)
