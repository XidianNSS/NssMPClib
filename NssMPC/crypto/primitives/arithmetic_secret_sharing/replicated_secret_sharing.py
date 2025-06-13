"""
Replicated Secret Sharing
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch

from NssMPC.common.ring import *
from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.primitives.arithmetic_secret_sharing._arithmetic_base import SecretSharingBase, RingPair
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import v_mul, v_matmul, truncate
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.compare import \
    secure_ge as v_secure_ge
from NssMPC.crypto.protocols.replicated_secret_sharing.semi_honest_functional import (secure_ge, mul_with_out_trunc,
                                                                                      matmul_with_out_trunc)
from NssMPC.secure_model.mpc_party import HonestMajorityParty


class ReplicatedSecretSharing(SecretSharingBase):
    """
    A class for replicated secret sharing over a RingPair.

    This class extends ArithmeticBase and provides methods for performing
    3PC(Secure 3-Party Computation) replicated secret sharing, as well as arithmetic operations (addition, subtraction,
    multiplication, etc.) **on 3PC secret-shared values**.

    :param ring_pair: The RingTensor pair(RingPair) used for secret sharing.
    :type ring_pair: RingPair(RingPair)

    .. note::
        The name of this class is abbreviated as **RSS** below.

    """

    def __init__(self, ring_pair):
        """
        Initializes an ReplicatedSecretSharing object.

        :param ring_pair: The RingTensor pair used for secret sharing.
        :type ring_pair: RingPair or list[RingTensor]
        """
        if isinstance(ring_pair, list):
            ring_pair = RingPair(ring_pair[0], ring_pair[1])
        super(ReplicatedSecretSharing, self).__init__(ring_pair)

    def __getitem__(self, item):
        """
        Enables indexing for the RSS instance, allowing elements to be accessed
        using square brackets [ ] like a list.

        :param item: The position of the element to retrieve.
        :type item: int
        :returns: The element at the specified index from the original RSS instance
        :rtype: ReplicatedSecretSharing

        """
        return ReplicatedSecretSharing([self.item[0][item], self.item[1][item]])

    def __setitem__(self, key, value):
        """
        Allows assignment to a specific index in the RSS instance
        using square brackets [ ].

        :param key: The position of the element to be set.
        :type key: int
        :param value: The value to assign to the specified index.
        :type value: ReplicatedSecretSharing
        :returns: None
        :raises TypeError: If `value` is not of a supported type(`RSS`).
        """
        if isinstance(value, ReplicatedSecretSharing):
            self.item[0][key] = value.item[0].clone()
            self.item[1][key] = value.item[1].clone()
        else:
            raise TypeError(f"unsupported operand type(s) for setitem '{type(self)}' and {type(value)}")

    def __str__(self):
        """
        Returns a string representation of the RSS instance.

        :returns: A string that represents the RSS instance.
        :rtype: str

        """
        return f"[{self.__class__.__name__}\n {self.item}]"

    def __add__(self, other):
        """
        Adds an RSS object with a corresponding type.

        :param other: The object to be added.
        :type other: ReplicatedSecretSharing or RingTensor or int
        :returns: A new RSS instance representing the result.
        :rtype: ReplicatedSecretSharing
        :raises TypeError: If `other` is not of a supported type.
        """
        party = PartyRuntime.party
        if isinstance(other, ReplicatedSecretSharing):
            return self.__class__(RingPair(self.item[0] + other.item[0], self.item[1] + other.item[1]))
        elif isinstance(other, RingTensor):
            zeros = RingTensor.zeros_like(other, dtype=other.dtype, device=other.device)
            if party.party_id == 0:
                return self.__class__(RingPair(self.item[0] + other, self.item[1] + zeros))
            elif party.party_id == 2:
                return self.__class__(RingPair(self.item[0] + zeros, self.item[1] + other))
            else:
                return self.__class__(RingPair(self.item[0] + zeros, self.item[1] + zeros))
        elif isinstance(other, (int, float)):
            other = RingTensor.convert_to_ring(int(other * self.scale))
            if party.party_id == 0:
                return self.__class__(RingPair(self.item[0] + other, self.item[1]))
            elif party.party_id == 2:
                return self.__class__(RingPair(self.item[0], self.item[1] + other))
            else:
                return self.clone()
        else:
            raise TypeError("unsupported operand type(s) for + 'ReplicatedSecretSharing' and ", type(other))

    __radd__ = __add__

    def __sub__(self, other):
        """
        Subtracts an RSS object with a corresponding type.

        :param other: The object to subtract.
        :type other: ReplicatedSecretSharing or RingTensor or int
        :returns: A new RSS instance representing the result.
        :rtype: ReplicatedSecretSharing
        :raises TypeError: If `other` is not of a supported type.
        """
        party = PartyRuntime.party
        if isinstance(other, ReplicatedSecretSharing):
            return self.__class__(RingPair(self.item[0] - other.item[0], self.item[1] - other.item[1]))
        elif isinstance(other, RingTensor):
            zeros = RingTensor.zeros_like(other, dtype=other.dtype, device=other.device)
            if party.party_id == 0:
                return self.__class__(RingPair(self.item[0] - other, self.item[1] - zeros))
            elif party.party_id == 2:
                return self.__class__(RingPair(self.item[0] - zeros, self.item[1] - other))
            else:
                return self.__class__(RingPair(self.item[0] - zeros, self.item[1] - zeros))
        elif isinstance(other, int):
            other = RingTensor.convert_to_ring(int(other * self.scale))
            if party.party_id == 0:
                return self.__class__(RingPair(self.item[0] - other, self.item[1]))
            elif party.party_id == 2:
                return self.__class__(RingPair(self.item[0], self.item[1] - other))
            else:
                return self.clone()
        else:
            raise TypeError("unsupported operand type(s) for - 'ReplicatedSecretSharing' and ", type(other))

    def __rsub__(self, other):
        return -(self - other)

    def __mul__(self, other):
        """
        Multiplies an RSS object with a corresponding type.
        If other is an RSS instance, it performs a multiplication using the v_mul method(ref:v_mul)
        for security of the computation.

        :param other: The object to be multiplied.
        :type other: ArithmeticSecretSharing or RingTensor or int.
        :returns: A new ASS instance representing the result.
        :rtype: ArithmeticSecretSharing
        :raises TypeError: If `other` is not of a supported type.

        .. note::
            Type 'float' is not supported.
        """
        if isinstance(other, ReplicatedSecretSharing):
            if isinstance(PartyRuntime.party, HonestMajorityParty):
                return v_mul(self, other)
            else:
                result = mul_with_out_trunc(self, other)
                if self.dtype == "float":
                    result = truncate(result)
                return result
        elif isinstance(other, RingTensor):
            result0 = RingTensor.mul(self.item[0], other)
            result1 = RingTensor.mul(self.item[1], other)
            result = ReplicatedSecretSharing(RingPair(result0, result1))
            if self.dtype == "float":
                result = truncate(result)
            return result
        elif isinstance(other, int):
            result = self.item * other
            return ReplicatedSecretSharing(result)
        else:
            raise TypeError("unsupported operand type(s) for * 'ReplicatedSecretSharing' and ", type(other))

    __rmul__ = __mul__

    def __matmul__(self, other):
        """
        Performs matrix multiplication between an RSS object with a corresponding type.

        If other is an RSS instance, it performs a matrix multiplication using
        the secure matrix multiplication protocol(ref:secure_matmul) for security of the computation.

        :param other: The object to perform matrix multiplication.
        :type other: ReplicatedSecretSharing or RingTensor
        :returns: A new RSS instance representing the result.
        :rtype: ReplicatedSecretSharing
        :raises TypeError: If `other` is not of a supported type.

        .. note::
            The input parameter type must be RSS or RingTensor.
        """
        if isinstance(other, ReplicatedSecretSharing):
            if isinstance(PartyRuntime.party, HonestMajorityParty):
                torch.cuda.empty_cache()
                return v_matmul(self, other)
            else:
                if self.device != other.device:
                    raise TypeError(
                        "Expected all ring tensors to be on the same device, but found at least two devices,"
                        + f" {self.device} and {other.device}!")

                result = matmul_with_out_trunc(self, other)
                if self.dtype == "float":
                    result = truncate(result)
                torch.cuda.empty_cache()
                return result
        elif isinstance(other, RingTensor):
            result0 = RingTensor.matmul(self.item[0], other)
            result1 = RingTensor.matmul(self.item[1], other)
            result = ReplicatedSecretSharing(RingPair(result0, result1))
            if self.dtype == "float":
                result = truncate(result)
            torch.cuda.empty_cache()
            return result
        else:
            raise TypeError("unsupported operand type(s) for @ 'ReplicatedSecretSharing' and ", type(other))

    def __rmatmul__(self, other):
        if isinstance(other, ReplicatedSecretSharing):
            if isinstance(PartyRuntime.party, HonestMajorityParty):
                torch.cuda.empty_cache()
                return v_matmul(other, self)
            else:
                if self.device != other.device:
                    raise TypeError(
                        "Expected all ring tensors to be on the same device, but found at least two devices,"
                        + f" {self.device} and {other.device}!")

                result = matmul_with_out_trunc(other, self)
                if self.dtype == "float":
                    result = truncate(result)
                torch.cuda.empty_cache()
                return result
        elif isinstance(other, RingTensor):
            result0 = RingTensor.matmul(other, self.item[0])
            result1 = RingTensor.matmul(other, self.item[1])
            result = ReplicatedSecretSharing(RingPair(result0, result1))
            if self.dtype == "float":
                result = truncate(result)
            torch.cuda.empty_cache()
            return result
        else:
            raise TypeError("unsupported operand type(s) for @ 'ReplicatedSecretSharing' and ", type(other))

    def __ge__(self, other):
        """
        Compare if the RSS instance is greater than or equal to `other`.

        This method compares the current instance with another RSS instance,
        a RingTensor, or an integer/float for a greater than or equal relationship with the secure_ge protocol(ref:secure_ge).

        :param other: The object to compare against.
        :type other: ReplicatedSecretSharing or RingTensor or int or float
        :returns: The corresponding element will be 1 if the two values are equal, otherwise 0.
        :rtype: ReplicatedSecretSharing
        :raises TypeError: If `other` is not of a supported type.
        """
        party = PartyRuntime.party
        if isinstance(party, HonestMajorityParty):
            ge = v_secure_ge
        else:
            ge = secure_ge
        if isinstance(other, ReplicatedSecretSharing):
            return ge(self, other)
        elif isinstance(other, RingTensor):
            zeros = RingTensor.zeros_like(other, dtype=other.dtype, device=other.device)
            if party.party_id == 0:
                other = ReplicatedSecretSharing([other, zeros])
            elif party.party_id == 1:
                other = ReplicatedSecretSharing([zeros, zeros])
            elif party.party_id == 2:
                other = ReplicatedSecretSharing([zeros, other])
            return ge(self, other)
        elif isinstance(other, int):
            return self >= RingTensor.convert_to_ring(int(other * self.scale))
        elif isinstance(other, float):
            assert self.dtype == 'float', "only float can compare with float"
            return self >= RingTensor.convert_to_ring(other)
        else:
            raise TypeError(f"unsupported operand type(s) for comparison '{type(self)}' and {type(other)}")

    def __le__(self, other):
        """
        Compare if the RSS instance is less than or equal to `other`.

        This method performs the comparison by negating the `__ge__` logic.

        :param other: The object to compare against.
        :type other: ReplicatedSecretSharing or RingTensor or int or float
        :returns: The corresponding element will be 1 if the two values are equal, otherwise 0.
        :rtype: ReplicatedSecretSharing
        """
        return -self >= -other

    def __gt__(self, other):
        """
        Compare if the RSS instance is greater than `other`.

        This method is based on the `__le__` comparison.

        :param other: The object to compare against.
        :type other: ReplicatedSecretSharing or RingTensor or int or float
        :returns: The corresponding element will be 1 if the two values are equal, otherwise 0.
        :rtype: ReplicatedSecretSharing
        """
        le = self <= other
        return -(le - 1)

    def __lt__(self, other):
        """
        Compare if the ArithmeticSecretSharing instance is less than `other`.

        This method is based on the `__ge__` comparison.

        :param other: The object to compare against.
        :type other: ReplicatedSecretSharing or RingTensor or int or float
        :returns: The corresponding element will be 1 if the two values are equal, otherwise 0.
        :rtype: ReplicatedSecretSharing
        """
        ge = self >= other
        return -(ge - 1)

    @classmethod
    def cat(cls, tensor_list, dim=0):
        """
        Concatenate a list of RSS instances along a given dimension.

        This method concatenates the `RingPair` objects inside each RSS
        instance in the list along the specified dimension.

        :param tensor_list: A list of RSS instances to concatenate.
        :type tensor_list: List[ReplicatedSecretSharing]
        :param dim: The dimension along which to concatenate the tensors. Defaults to 0.
        :type dim: int
        :returns: A new RSS instance with concatenated RingTensors.
        :rtype: ReplicatedSecretSharing
        """
        result_0 = RingTensor.cat([e.item[0] for e in tensor_list], dim)
        result_1 = RingTensor.cat([e.item[1] for e in tensor_list], dim)
        return cls(RingPair(result_0, result_1))

    @classmethod
    def stack(cls, tensor_list, dim=0):
        """
        Stack a list of RSS instances along a new dimension.

        This method stacks the `RingPair` objects inside each RSS
        instance in the list along a new dimension specified by `dim`.

        :param tensor_list: A list of RSS instances to stack.
        :type tensor_list: List[ReplicatedSecretSharing]
        :param dim: The dimension along which to stack the tensors. Defaults to 0.
        :type dim: int
        :returns: A new RSS instance with stacked tensors.
        :rtype: ReplicatedSecretSharing
        """
        result_0 = RingTensor.stack([e.item[0] for e in tensor_list], dim)
        result_1 = RingTensor.stack([e.item[1] for e in tensor_list], dim)
        return cls(RingPair(result_0, result_1))

    @classmethod
    def roll(cls, input, shifts, dims=0):
        """
        Roll the elements of an RSS instance along a specified dimension.

        This method rolls the elements of the `RingPair` in the specified `input` along
        the `dims` dimension by the number of positions specified in `shifts`.

        :param input: The RSS instance whose elements will be rolled.
        :type input: ReplicatedSecretSharing
        :param shifts: The number of positions by which the elements are shifted.
        :type shifts: int
        :param dims: The dimension along which to roll the elements. Defaults to 0.
        :type dims: int
        :returns: A new RSS instance with the rolled elements.
        :rtype: ReplicatedSecretSharing
        """
        result_0 = RingTensor.roll(input.item[0], shifts, dims)
        result_1 = RingTensor.roll(input.item[1], shifts, dims)
        return cls(RingPair(result_0, result_1))

    @classmethod
    def rotate(cls, input, shifts):
        """
        Rotate each row of a 2-dimensional RSS instance to the left or right.

        This method rotates each row of the 2D `RingPair` in the specified `input` by the
        number of positions specified in `shifts`.

        :param input: The 2-dimensional ASS instance to rotate.
        :type input: ReplicatedSecretSharing
        :param shifts: The number of positions to rotate each row. Positive values rotate to the right, negative to the left.
        :type shifts: int
        :returns: A new RSS instance with the rotated rows.
        :rtype: ReplicatedSecretSharing
        """
        result_0 = RingTensor.rotate(input.item[0], shifts)
        result_1 = RingTensor.rotate(input.item[1], shifts)
        return cls(RingPair(result_0, result_1))

    @staticmethod
    def gen_and_share(r_tensor, party):
        """
        Generate and share the input r_tensor.

        Share the r_tenor to another two parties, and return one share of the r_tensor which
        should be kept by the current party.

        :param r_tensor: The RingTensor to be shared.
        :type r_tensor: RingTensor
        :param party: The party that holds the RingPair.
        :type party: Party

        :returns: The share should be kept by the current party.
        :rtype: ReplicatedSecretSharing

        """
        r0, r1, r2 = ReplicatedSecretSharing.share(r_tensor)
        party.send((party.party_id + 1) % 3, r1)
        party.send((party.party_id + 2) % 3, r2)
        return r0

    @classmethod
    def share(cls, tensor: RingTensor):
        """
        Perform a three-party replicated secret sharing on the input tensor.

        :param tensor: The RingTensor to be shared.
        :type tensor: RingTensor
        :returns: The list of three RSS shares, and the attribute——Party of each share is None.
        :rtype: List[ReplicatedSecretSharing]
        """
        shares = []
        x_0 = RingTensor.random(tensor.shape, tensor.dtype, tensor.device)
        x_1 = RingTensor.random(tensor.shape, tensor.dtype, tensor.device)
        x_2 = tensor - x_0 - x_1
        shares.append(ReplicatedSecretSharing([x_0, x_1]))
        shares.append(ReplicatedSecretSharing([x_1, x_2]))
        shares.append(ReplicatedSecretSharing([x_2, x_0]))
        return shares

    def restore(self):
        """
        Restore the original data from the secret shares.

        Perform the send-and-receive process to restore the original RingTensor.

        :returns: The restored RingTensor.
        :rtype: RingTensor
        """
        # 发送部分
        party = PartyRuntime.party
        party.send((party.party_id + 1) % 3, self.item[0])
        # 接收部分
        other = party.receive((party.party_id + 2) % 3)
        return self.item[0] + self.item[1] + other

    @staticmethod
    def reshare(value: RingTensor, party):
        """
        Reshare the shared values held by each party without altering the restored value.

        We use this method to convert the state where each party holds a single share
        into a state where they hold RSS shares, i.e., RingPair.

        :param value: The share that the current party holds.
        :type value: RingTensor
        :param party: The party that holds the share.
        :type party: Party
        :returns: The RSS instance that belongs to the current party.
        :rtype: ReplicatedSecretSharing
        """
        r_0 = party.prg_0.random(value.numel())
        r_1 = party.prg_1.random(value.numel())
        r_0 = r_0.reshape(value.shape)
        r_1 = r_1.reshape(value.shape)
        r_0.dtype = r_1.dtype = value.dtype
        value = value.tensor + r_0 - r_1
        party.send((party.party_id + 2) % 3, value)
        other = party.receive((party.party_id + 1) % 3)
        return ReplicatedSecretSharing(RingPair(value, other))

    @classmethod
    def random(cls, shape, party):
        """
        Generate RSS with a specified shape and random content.

        We use this method to generate a random RSS instance with a specified shape and party.

        :param shape: The tuple representing the desired shape of the RSS.
        :type shape: tuple or list
        :param party: The party that holds the random RSS.
        :type party: Party
        :returns: The RSS instance with the specified shape and random content.
        :rtype: ReplicatedSecretSharing
        """
        num_of_value = 1
        for d in shape:
            num_of_value *= d
        r_0 = party.prg_0.random(num_of_value)
        r_1 = party.prg_1.random(num_of_value)
        r = ReplicatedSecretSharing([r_0, r_1])
        r = r.reshape(shape)
        return r

    @classmethod
    def rand_like(cls, x, party):
        """
        Generate a random RSS with the same shape as input *x*.

        We use this method to generate a random RSS with the same shape as input *x*.

        :param x: The input whose shape will be used to generate the random RSS.
        :type x: ReplicatedSecretSharing, RingTensor or torch.Tensor
        :param party: The party that holds the random RSS.
        :type party: Party
        :returns: The RSS instance with the same shape as *x*.
        :rtype: ReplicatedSecretSharing
        """
        r = ReplicatedSecretSharing.random(x.shape, party)
        if isinstance(x, (RingTensor, ReplicatedSecretSharing)):
            r.dtype = x.dtype
        return r
