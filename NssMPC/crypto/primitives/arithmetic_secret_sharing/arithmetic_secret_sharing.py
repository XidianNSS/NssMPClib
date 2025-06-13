"""Arithmetic Secret Sharing
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.common.ring import *
from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.primitives.arithmetic_secret_sharing._arithmetic_base import SecretSharingBase, auto_delegate
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional import beaver_mul, secure_matmul, \
    secure_div, secure_eq, secure_ge, secure_exp, secure_reciprocal_sqrt, truncate, secure_tanh


class ArithmeticSecretSharing(SecretSharingBase):
    """
    A class for arithmetic secret sharing over a RingTensor.

    This class extends ArithmeticBase and provides methods for performing
    secret sharing, as well as arithmetic operations (addition, subtraction,
    multiplication, etc.) **on secret-shared values**.

    :param ring_tensor: The tensor used for secret sharing.
    :type ring_tensor: RingTensor

    .. note::
        The name of this class is abbreviated as **ASS** below.

    """

    def __init__(self, ring_tensor):
        """
        Initializes an ASS object.

        :param ring_tensor: The RingTensor used for secret sharing.
        :type ring_tensor: RingTensor
        """
        assert isinstance(ring_tensor, RingTensor), "ring_tensor must be a RingTensor"
        super().__init__(ring_tensor)

    @property
    def ring_tensor(self):
        """Get ring_tensor from an ASS instance."""
        return self.item

    def __getitem__(self, item):
        """
        Enables indexing for the ASS instance, allowing elements to be accessed
        using square brackets [ ] like a list.

        :param item: The position of the element to retrieve.
        :type item: int
        :returns: The element at the specified index from the original ASS instance
        :rtype: ArithmeticSecretSharing

        """
        return ArithmeticSecretSharing(self.item[item])

    def __setitem__(self, key, value):
        """
        Allows assignment to a specific index in the ASS instance
        using square brackets [ ].

        :param key: The position of the element to be set.
        :type key: int
        :param value: The value to assign to the specified index.
        :type value: ArithmeticSecretSharing
        :returns: None
        :raises TypeError: If `value` is not of a supported type(`ASS`).
        """
        if isinstance(value, ArithmeticSecretSharing):
            self.item[key] = value.item.clone()
        else:
            raise TypeError(f"unsupported operand type(s) for setitem '{type(self)}' and {type(value)}")

    def __add__(self, other):
        if isinstance(other, ArithmeticSecretSharing):
            new_tensor = self.item + other.item
            return ArithmeticSecretSharing(new_tensor)
        elif isinstance(other, RingTensor):  # for RingTensor or const number, only party 0 add it to the share tensor
            if PartyRuntime.party.party_id == 0:
                new_tensor = self.item + other
            else:
                new_tensor = self.item + RingTensor.zeros_like(other)
            return ArithmeticSecretSharing(new_tensor)
        elif isinstance(other, (int, float)):
            other = RingTensor.convert_to_ring(int(other * self.scale))
            return self + other
        else:
            raise TypeError(f"unsupported operand type(s) for + '{type(self)}' and {type(other)}")

    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, ArithmeticSecretSharing):
            self.item += other.item
        elif isinstance(other, RingTensor):
            if PartyRuntime.party.party_id == 0:
                self.item += other
            else:
                self.item += RingTensor.zeros_like(other)
        elif isinstance(other, (int, float)):
            self += RingTensor.convert_to_ring(int(other * self.scale))
        else:
            raise TypeError(f"unsupported operand type(s) for += '{type(self)}' and {type(other)}")
        return self

    def __sub__(self, other):
        if isinstance(other, ArithmeticSecretSharing):
            new_tensor = self.item - other.item
            return ArithmeticSecretSharing(new_tensor)
        elif isinstance(other, RingTensor):
            if PartyRuntime.party.party_id == 0:
                new_tensor = self.item - other
            else:
                new_tensor = self.item - RingTensor.zeros_like(other)
            return ArithmeticSecretSharing(new_tensor)
        elif isinstance(other, (int, float)):
            other = RingTensor.convert_to_ring(int(other * self.scale))
            return self - other
        else:
            raise TypeError(f"unsupported operand type(s) for - '{type(self)}' and {type(other)}")

    def __rsub__(self, other):
        return -(self - other)

    def __isub__(self, other):
        if isinstance(other, ArithmeticSecretSharing):
            self.item -= other.item
        elif isinstance(other, RingTensor):
            if PartyRuntime.party.party_id == 0:
                self.item -= other
            else:
                self.item -= RingTensor.zeros_like(other)
        elif isinstance(other, (int, float)):
            self -= RingTensor.convert_to_ring(int(other * self.scale))
        else:
            raise TypeError(f"unsupported operand type(s) for -= '{type(self)}' and {type(other)}")
        return self

    def __mul__(self, other):
        """
        Multiplies an ASS object with a corresponding type.

        If other is an ASS instance, it performs a multiplication using :meth:`~NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional.multiplication.beaver_mul`.

        :param other: The object to be multiplied.
        :type other: ArithmeticSecretSharing or RingTensor or int.
        :returns: A new ASS instance representing the result.
        :rtype: ArithmeticSecretSharing
        :raises TypeError: If `other` is not of a supported type.

        .. note::
            Type 'float' is not supported.
        """
        if isinstance(other, ArithmeticSecretSharing):
            res = beaver_mul(self, other)
        elif isinstance(other, RingTensor):
            res = ArithmeticSecretSharing(RingTensor.mul(self.item, other))
        elif isinstance(other, int):
            return ArithmeticSecretSharing(self.item * other)
        elif isinstance(other, float):
            assert self.dtype == 'float', "only float can calculate with float"
            res = ArithmeticSecretSharing(self.item * int(other * self.scale))
        else:
            raise TypeError(f"unsupported operand type(s) for * '{type(self)}' and {type(other)}")

        return res / self.scale

    __rmul__ = __mul__

    def __matmul__(self, other):
        """
        Performs matrix multiplication between an ASS object with a corresponding type.

        If other is an ASS instance, it performs a matrix multiplication using
        the secure matrix multiplication protocol :meth:`~NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional.multiplication.secure_matmul` for security of the computation.

        :param other: The object to perform matrix multiplication.
        :type other: ArithmeticSecretSharing or RingTensor
        :returns: A new ASS instance representing the result.
        :rtype: ArithmeticSecretSharing
        :raises TypeError: If `other` is not of a supported type.

        .. note::
            The input parameter type must be ASS or RingTensor.
        """
        if isinstance(other, ArithmeticSecretSharing):
            res = secure_matmul(self, other)
        elif isinstance(other, RingTensor):
            res = ArithmeticSecretSharing(RingTensor.matmul(self.item, other))
        else:
            raise TypeError(f"unsupported operand type(s) for @ '{type(self)}' and {type(other)}")

        return res / self.scale

    def __rmatmul__(self, other):
        if isinstance(other, ArithmeticSecretSharing):
            res = secure_matmul(other, self)
        elif isinstance(other, RingTensor):
            res = ArithmeticSecretSharing(RingTensor.matmul(other, self.item))
        else:
            raise TypeError(f"unsupported operand type(s) for @ '{type(self)}' and {type(other)}")

        return res / self.scale

    def __pow__(self, power, modulo=None):
        """
        Raise the ASS instance to the power of `power`.

        This method performs exponentiation on the current instance using the
        given `power`. Optionally, results can be reduced modulo `modulo` if provided.

        :param int power: The exponent to which the instance will be raised.
        :param int modulo: The modulo value to reduce the result. Defaults to None.
        :returns: A new ASS instance representing the result of exponentiation.
        :rtype: ArithmeticSecretSharing
        :raises TypeError: If `power` is not an integer.
        """
        # TODO: 'continue' coming soon.
        if isinstance(power, int):
            temp = self
            res = 1
            while power:
                if power % 2 == 1:
                    res = temp * res
                temp = temp * temp
                power //= 2
            return res
        else:
            raise TypeError(f"unsupported operand type(s) for ** '{type(self)}' and {type(power)}")

    def __truediv__(self, other):
        """
        Divide the ASS instance by `other`.

        This method supports division by an integer, float, another ASS
        instance, or a RingTensor. If the other is an ASS instance, it performs a division using
        the secure division protocol :meth:`~NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional.division.secure_div` for security of the computation.

        :param other: The value by which to divide. Can be an int or float or ArithmeticSecretSharing or RingTensor.
        :type other: int or float or ArithmeticSecretSharing or RingTensor
        :returns: A new ASS instance representing the result of division.
        :rtype: ArithmeticSecretSharing
        :raises TypeError: If `other` is not of a supported type.
        """
        if isinstance(other, int):
            if other == 1:
                return self
            return truncate(self, other)
        elif isinstance(other, float):
            if other == 1:
                return self
            from NssMPC.config import float_scale
            return (self / int(other * float_scale)) * float_scale
        elif isinstance(other, ArithmeticSecretSharing):
            return secure_div(self, other)
        elif isinstance(other, RingTensor):
            return truncate(self * other.scale, other.tensor)
        else:
            raise TypeError(f"unsupported operand type(s) for / '{type(self)}' and {type(other)}")

    def __eq__(self, other):
        """
        Compare the ASS instance for equality with `other`.

        This method compares the current instance with another ASS instance,
        a RingTensor, or an integer/float with the secure equality protocol :meth:`~NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional.comparison.secure_eq`.

        :param other: The object to compare for equality.
        :type other: ArithmeticSecretSharing or RingTensor or int or float
        :returns: The corresponding element will be 1 if the two values are equal, otherwise 0.
        :rtype: ArithmeticSecretSharing
        :raises TypeError: If `other` is not of a supported type.
        """
        if isinstance(other, (ArithmeticSecretSharing, RingTensor)):
            return secure_eq(self, other)
        elif isinstance(other, int):
            return secure_eq(self, RingTensor.convert_to_ring(int(other * self.scale)))
        elif isinstance(other, float):
            assert self.dtype == 'float', "only float can compare with float"
            return secure_eq(self, RingTensor.convert_to_ring(other))
        else:
            raise TypeError(f"unsupported operand type(s) for comparison '{type(self)}' and {type(other)}")

    def __ge__(self, other):
        """
        Compare if the ASS instance is greater than or equal to `other`.

        This method compares the current instance with another ASS instance,
        a RingTensor, or an integer/float for a greater than or equal relationship with the secure_ge protocol :meth:`~NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional.comparison.secure_ge`.

        :param other: The object to compare against.
        :type other: ArithmeticSecretSharing or RingTensor or int or float
        :returns: The corresponding element will be 1 if the two values are equal, otherwise 0.
        :rtype: ArithmeticSecretSharing
        :raises TypeError: If `other` is not of a supported type.
        """
        if isinstance(other, (ArithmeticSecretSharing, RingTensor)):
            return secure_ge(self, other)
        elif isinstance(other, int):
            return secure_ge(self, RingTensor.convert_to_ring(int(other * self.scale)))
        elif isinstance(other, float):
            assert self.dtype == 'float', "only float can compare with float"
            return secure_ge(self, RingTensor.convert_to_ring(other))
        else:
            raise TypeError(f"unsupported operand type(s) for comparison '{type(self)}' and {type(other)}")

    def __le__(self, other):
        """
        Compare if the ASS instance is less than or equal to `other`.

        This method performs the comparison by negating the `__ge__` logic.

        :param other: The object to compare against.
        :type other: ArithmeticSecretSharing or RingTensor or int or float
        :returns: The corresponding element will be 1 if the two values are equal, otherwise 0.
        :rtype: ArithmeticSecretSharing
        """
        return -self >= -other

    def __gt__(self, other):
        """
        Compare if the ArithmeticSecretSharing instance is greater than `other`.

        This method is based on the `__le__` comparison.

        :param other: The object to compare against.
        :type other: ArithmeticSecretSharing or RingTensor or int or float
        :returns: The corresponding element will be 1 if the two values are equal, otherwise 0.
        :rtype: ArithmeticSecretSharing
        """
        le = self <= other
        return -(le - 1)

    def __lt__(self, other):
        """
        Compare if the ArithmeticSecretSharing instance is less than `other`.

        This method is based on the `__ge__` comparison.

        :param other: The object to compare against.
        :type other: ArithmeticSecretSharing or RingTensor or int or float
        :returns: The corresponding element will be 1 if the two values are equal, otherwise 0.
        :rtype: ArithmeticSecretSharing
        """
        ge = self >= other
        return -(ge - 1)

    @classmethod
    def cat(cls, tensor_list, dim=0):
        """
        Concatenate a list of ASS instances along a given dimension.

        This method concatenates the `RingTensor` objects inside each ASS
        instance in the list along the specified dimension.

        :param tensor_list: A list of ASS instances to concatenate.
        :type tensor_list: List[ArithmeticSecretSharing]
        :param dim: The dimension along which to concatenate the tensors. Defaults to 0.
        :type dim: int
        :returns: A new ASS instance with concatenated RingTensors.
        :rtype: ArithmeticSecretSharing
        """
        result = RingTensor.cat([e.item for e in tensor_list], dim)
        return cls(result)

    @classmethod
    def stack(cls, tensor_list, dim=0):
        """
        Stack a list of ASS instances along a new dimension.

        This method stacks the `RingTensor` objects inside each ASS
        instance in the list along a new dimension specified by `dim`.

        :param tensor_list: A list of ASS instances to stack.
        :type tensor_list: List[ArithmeticSecretSharing]
        :param dim: The dimension along which to stack the tensors. Defaults to 0.
        :type dim: int
        :returns: A new ASS instance with stacked tensors.
        :rtype: ArithmeticSecretSharing
        """
        result = RingTensor.stack([e.item for e in tensor_list], dim)
        return cls(result)

    @classmethod
    def gather(cls, input, dim, index):
        """
        Gather elements from the ASS instance along an axis specified by `dim`.

        This method gathers elements from `input` along the specified dimension `dim`,
        using the indices provided by `index`.

        :param input: The ASS instance from which to gather elements.
        :type input: ArithmeticSecretSharing
        :param dim: The dimension along which to gather.
        :type dim: int
        :param index: The indices of the elements to gather.
        :type index: Tensor
        :returns: A new ASS instance with gathered elements.
        :rtype: ArithmeticSecretSharing
        """
        result = RingTensor.gather(input.item, dim, index)
        return cls(result)

    @classmethod
    def roll(cls, input, shifts, dims=0):
        """
        Roll the elements of an ASS instance along a specified dimension.

        This method rolls the elements of the `RingTensor` in the specified `input` along
        the `dims` dimension by the number of positions specified in `shifts`.

        :param input: The ASS instance whose elements will be rolled.
        :type input: ArithmeticSecretSharing
        :param shifts: The number of positions by which the elements are shifted.
        :type shifts: int
        :param dims: The dimension along which to roll the elements. Defaults to 0.
        :type dims: int
        :returns: A new ASS instance with the rolled elements.
        :rtype: ArithmeticSecretSharing
        """
        result = RingTensor.roll(input.item, shifts, dims)
        return cls(result)

    @classmethod
    def rotate(cls, input, shifts):
        """
        Rotate each row of a 2-dimensional ASS instance to the left or right.

        This method rotates each row of the 2D `RingTensor` in the specified `input` by the
        number of positions specified in `shifts`.

        :param input: The 2-dimensional ASS instance to rotate.
        :type input: ArithmeticSecretSharing
        :param shifts: The number of positions to rotate each row. Positive values rotate to the right, negative to the left.
        :type shifts: int
        :returns: A new ASS instance with the rotated rows.
        :rtype: ArithmeticSecretSharing
        """
        result = RingTensor.rotate(input.item, shifts)
        return cls(result)

    @classmethod
    def exp(cls, x):
        """
        Compute the element-wise exponential of an ASS instance.

        This method applies the exponential function element-wise on the given
        ASS instance using secure_exp method for secure computation.

        :param x: The ASS instance to which the exponential function is applied.
        :type x: ArithmeticSecretSharing
        :returns: A new ASS instance with the exponential applied to each element.
        :rtype: ArithmeticSecretSharing
        """
        return secure_exp(x)

    @classmethod
    def rsqrt(cls, x):
        """
        Compute the element-wise reciprocal square root of an ASS instance.

        This method applies the reciprocal square root function element-wise on the given
        ASS instance using secure_reciprocal_sqrt method for secure computation.

        :param x: The ASS instance to which the reciprocal square root function is applied.
        :type x: ArithmeticSecretSharing
        :returns: A new ASS instance with the reciprocal square root applied to each element.
        :rtype: ArithmeticSecretSharing
        """
        return secure_reciprocal_sqrt(x)

    @classmethod
    def tanh(cls, x):
        """
        Compute the element-wise hyperbolic tangent (tanh) of an ASS instance.

        This method applies the tanh function element-wise on the given
        ASS instance using secure_tanh method for secure computation.

        :param x: The ASS instance to which the tanh function is applied.
        :type x: ArithmeticSecretSharing
        :returns: A new ASS instance with the tanh function applied to each element.
        :rtype: ArithmeticSecretSharing
        """
        return secure_tanh(x)

    @staticmethod
    def restore_from_shares(share_0, share_1):
        """
        Recovers the original data from two ASS instance(or we could say two shares).

        :param share_0: The first share.
        :type share_0: ArithmeticSecretSharing
        :param share_1: The second share.
        :type share_1: ArithmeticSecretSharing
        :returns: The original data recovered from the two shares.
        :rtype: RingTensor
        """
        return share_0.item + share_1.item

    def restore(self):
        """
         Restore the original data from secret shares.

         This method is used to reconstruct the original data by combining the shares
         held by different parties. It requires communication between the parties.

         :returns: The reconstructed original data as a RingTensor.
         :rtype: RingTensor
         """
        PartyRuntime.party.send(self)
        other = PartyRuntime.party.receive()
        return self.item + other.item

    @classmethod
    def share(cls, tensor: RingTensor, num_of_party: int = 2):
        """
        Perform secret sharing on a RingTensor across multiple parties.

        This method divides the given `RingTensor` into secret shares, distributing
        the shares among the specified number of parties. The returned shares
        are still RingTensors, not ASS objects.

        :param tensor: The RingTensor to be secret-shared.
        :type tensor: RingTensor
        :param num_of_party: The number of parties among which the tensor will be shared. Defaults to 2.
        :type num_of_party: int
        :returns: A list of ArithmeticSecretSharing shares, one for each party.
        :rtype: List[ArithmeticSecretSharing]
        """
        share = []
        x_0 = tensor.clone()

        for i in range(num_of_party - 1):
            x_i = RingTensor.random(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            share.append(cls(x_i))
            x_0 -= x_i
        share.append(cls(x_0))
        return share
