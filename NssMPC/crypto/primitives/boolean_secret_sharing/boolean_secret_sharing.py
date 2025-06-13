#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.common.ring import RingTensor
from NssMPC.config import DEVICE
from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.protocols.boolean_secret_sharing.semi_honest_functional.semi_honest_functional import beaver_and
from NssMPC.common.ring import *
from NssMPC.crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import SecretSharingBase


class BooleanSecretSharing(SecretSharingBase):
    """

    A class for Boolean secret sharing, supporting various operations over boolean secret shared RingTensor.

    This class extends ArithmeticBase and provides methods for performing
    boolean secret sharing, as well as boolean operations (invert, and, etc.) **on secret-shared values**.

    :param ring_tensor: The RingTensor used for boolean secret sharing.
    :type ring_tensor: RingTensor
    :raises AssertionError: If `ring_tensor` is not an instance of `RingTensor`.

    .. note::
        The name of this class is abbreviated as **BSS** below.


    """

    def __init__(self, ring_tensor):
        """
        Initializes an BSS object.

        :param ring_tensor: The RingTensor used for boolean secret sharing.
        :type ring_tensor: RingTensor
        """
        assert isinstance(ring_tensor, RingTensor), "ring_tensor must be a RingTensor"
        super().__init__(ring_tensor)

    @property
    def ring_tensor(self):
        """Get ring_tensor from an BSS instance."""
        return self.item

    def __invert__(self):
        """
        Bitwise NOT operation on the BSS instance.

        :return: A new `BooleanSecretSharing` object with the inverted RingTensor.
        :rtype: BooleanSecretSharing
        """
        party = PartyRuntime.party
        if party.party_id == 0:
            new_tensor = ~self.item
            return BooleanSecretSharing(new_tensor)
        else:
            return BooleanSecretSharing(self.item)

    def __and__(self, other):
        """
        Bitwise AND operation between two BSS instances.

        When `other` is a BSS instance, the method uses the
        :func:`Beaver triples <NssMPC.crypto.protocols.boolean_secret_sharing.semi_honest_functional.semi_honest_functional.beaver_and>`
        **to ensure secure computation** during the bitwise AND operation. If `other` is a constant (i.e., a RingTensor),
        the method directly performs the bitwise AND operation without additional security.

        :param other: Another BSS instance or RingTensor.
        :type other: BooleanSecretSharing or RingTensor
        :return: A new `BooleanSecretSharing` object representing the result.
        :rtype: BooleanSecretSharing
        :raises TypeError: If the type of `other` is unsupported.
        """
        if isinstance(other, BooleanSecretSharing):
            new_tensor = beaver_and(self, other)
            return BooleanSecretSharing(new_tensor)
        elif isinstance(other, RingTensor):
            return BooleanSecretSharing(self.item & other)
        else:
            raise TypeError(f"unsupported operand type(s) for + '{type(self)}' and {type(other)}")

    @classmethod
    def cat(cls, tensor_list, dim=0):
        """
        Concatenate a list of BSS instances along a given dimension.

        This method concatenates the `RingTensor` objects inside each BSS
        instance in the list along the specified dimension.

        :param tensor_list: A list of BSS instances to concatenate.
        :type tensor_list: List[BooleanSecretSharing]
        :param dim: The dimension along which to concatenate the tensors. Defaults to 0.
        :type dim: int
        :returns: A new BSS instance with concatenated RingTensors.
        :rtype: BooleanSecretSharing
        """
        result = RingTensor.cat([e.item for e in tensor_list], dim)
        return cls(result)

    @classmethod
    def stack(cls, tensor_list, dim=0):
        """
        Stack a list of BSS instances along a new dimension.

        This method stacks the `RingTensor` objects inside each BSS
        instance in the list along a new dimension specified by `dim`.

        :param tensor_list: A list of BSS instances to stack.
        :type tensor_list: List[BooleanSecretSharing]
        :param dim: The dimension along which to stack the tensors. Defaults to 0.
        :type dim: int
        :returns: A new BSS instance with stacked tensors.
        :rtype: BooleanSecretSharing
        """
        result = RingTensor.stack([e.item for e in tensor_list], dim)
        return cls(result)

    @classmethod
    def roll(cls, input, shifts, dims=0):
        """
        Roll the elements of an BSS instance along a specified dimension.

        This method rolls the elements of the `RingTensor` in the specified `input` along
        the `dims` dimension by the number of positions specified in `shifts`.

        :param input: The BSS instance whose elements will be rolled.
        :type input: BooleanSecretSharing
        :param shifts: The number of positions by which the elements are shifted.
        :type shifts: int
        :param dims: The dimension along which to roll the elements. Defaults to 0.
        :type dims: int
        :returns: A new BSS instance with the rolled elements.
        :rtype: BooleanSecretSharing
        """
        result = RingTensor.roll(input.item, shifts, dims)
        return cls(result)

    @classmethod
    def rotate(cls, input, shifts):
        """
        Rotate each row of a 2-dimensional BSS instance to the left or right.

        This method rotates each row of the 2D `RingTensor` in the specified `input` by the
        number of positions specified in `shifts`.

        :param input: The 2-dimensional BSS instance to rotate.
        :type input: BooleanSecretSharing
        :param shifts: The number of positions to rotate each row. Positive values rotate to the right, negative to the left.
        :type shifts: int
        :returns: A new BSS instance with the rotated rows.
        :rtype: BooleanSecretSharing
        """
        result = RingTensor.rotate(input.item, shifts)
        return cls(result)

    @staticmethod
    def restore_from_shares(share_0, share_1):
        """
        Recovers the original data from two BSS instance(or we could say two shares).

        :param share_0: The first share.
        :type share_0: BooleanSecretSharing
        :param share_1: The second share.
        :type share_1: BooleanSecretSharing
        :returns: The original data recovered from the two shares.
        :rtype: RingTensor
        """
        return share_0.item ^ share_1.item

    def restore(self):
        """
         Restore the original data from secret shares.

         This method is used to reconstruct the original data by combining the shares
         held by different parties. It requires communication between the parties.
         (ref: Communication between the parties.)

         :returns: The reconstructed original data as a RingTensor.
         :rtype: RingTensor
         """
        party = PartyRuntime.party
        party.send(self)
        other = party.receive()
        return self.item ^ other.item

    @staticmethod
    def share(x, num_of_party: int):
        """
        Perform secret sharing on a RingTensor across multiple parties in a bitwise manner.

        This method splits the input `x` into secret shares in bitwise, distributing
        the shares among the specified number of parties. The returned shares
        are still RingTensors, not BSS objects.

        :param x: The RingTensor to be secret-shared.
        :type x: RingTensor
        :param num_of_party: The number of parties to distribute the shares.
        :type num_of_party: int
        :returns: A list of RingTensor shares, one for each party.
        :rtype: List[RingTensor]

        .. note::
            The ring size of RingTensor in this method is 2, which means that the elements in the RingTensor are either
            0 or 1.

        """
        share_x = []
        x_0 = x.clone()

        for i in range(num_of_party - 1):
            x_i = RingTensor.random(x.shape, down_bound=0, upper_bound=2, device=DEVICE)
            share_x.append(x_i)
            x_0 ^= x_i
        share_x.append(x_0)
        return share_x
