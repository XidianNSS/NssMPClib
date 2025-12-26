#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from nssmpc.config import DEVICE
from nssmpc.infra.mpc.party import PartyCtx
from nssmpc.infra.tensor import RingTensor
from nssmpc.primitives.secret_sharing.arithmetic import SecretSharingScheme


class BooleanSecretSharing(SecretSharingScheme):
    """
    A class for Boolean secret sharing, supporting various operations over boolean secret shared RingTensor.

    This class extends ArithmeticBase and provides methods for performing
    boolean secret sharing, as well as boolean operations (invert, and, etc.) **on secret-shared values**.

    Note:
        The name of this class is abbreviated as **BSS** below.

    Args:
        ring_tensor (RingTensor): The RingTensor used for boolean secret sharing.

    Raises:
        AssertionError: If `ring_tensor` is not an instance of `RingTensor`.
    """

    def __init__(self, ring_tensor):
        """
        Initializes an BSS object.

        Args:
            ring_tensor (RingTensor): The RingTensor used for boolean secret sharing.

        Examples:
            >>> bss = BooleanSecretSharing(ring_tensor)
        """
        assert isinstance(ring_tensor, RingTensor), "ring_tensor must be a RingTensor"
        self.item = ring_tensor

    @property
    def ring_tensor(self):
        """
        Get ring_tensor from an BSS instance.

        Returns:
            RingTensor: The underlying RingTensor.

        Examples:
            >>> tensor = bss.ring_tensor
        """
        return self.item

    def __invert__(self):
        """
        Bitwise NOT operation on the BSS instance.

        Returns:
            BooleanSecretSharing: A new `BooleanSecretSharing` object with the inverted RingTensor.

        Examples:
            >>> inv_bss = ~bss
        """
        party = PartyCtx.current
        if party.party_id == 0:
            new_tensor = ~self.item
            return BooleanSecretSharing(new_tensor)
        else:
            return BooleanSecretSharing(self.item)

    def __and__(self, other):
        """
        Bitwise AND operation between two BSS instances.

        When `other` is a BSS instance, the method uses the
        :func:`Beaver triples <nssmpc.crypto.protocols.boolean_secret_sharing.semi_honest_functional.semi_honest_functional.beaver_and>`
        **to ensure secure computation** during the bitwise AND operation. If `other` is a constant (i.e., a RingTensor),
        the method directly performs the bitwise AND operation without additional security.

        Args:
            other (BooleanSecretSharing or RingTensor): Another BSS instance or RingTensor.

        Returns:
            BooleanSecretSharing: A new `BooleanSecretSharing` object representing the result.

        Raises:
            TypeError: If the type of `other` is unsupported.

        Examples:
            >>> res = bss1 & bss2
        """
        if isinstance(other, BooleanSecretSharing):
            new_tensor = self._and(other)
            return BooleanSecretSharing(new_tensor)
        elif isinstance(other, RingTensor):
            return BooleanSecretSharing(self.item & other)
        else:
            raise TypeError(f"unsupported operand type(s) for + '{type(self)}' and {type(other)}")

    def _and(self, other):
        raise NotImplementedError()

    @classmethod
    def cat(cls, tensor_list, dim=0):
        """
        Concatenate a list of BSS instances along a given dimension.

        This method concatenates the `RingTensor` objects inside each BSS
        instance in the list along the specified dimension.

        Args:
            tensor_list (List[BooleanSecretSharing]): A list of BSS instances to concatenate.
            dim (int, optional): The dimension along which to concatenate the tensors. Defaults to 0.

        Returns:
            BooleanSecretSharing: A new BSS instance with concatenated RingTensors.

        Examples:
            >>> res = BooleanSecretSharing.cat([bss1, bss2], dim=0)
        """
        result = RingTensor.cat([e.item for e in tensor_list], dim)
        return cls(result)

    @classmethod
    def stack(cls, tensor_list, dim=0):
        """
        Stack a list of BSS instances along a new dimension.

        This method stacks the `RingTensor` objects inside each BSS
        instance in the list along a new dimension specified by `dim`.

        Args:
            tensor_list (List[BooleanSecretSharing]): A list of BSS instances to stack.
            dim (int, optional): The dimension along which to stack the tensors. Defaults to 0.

        Returns:
            BooleanSecretSharing: A new BSS instance with stacked tensors.

        Examples:
            >>> res = BooleanSecretSharing.stack([bss1, bss2], dim=0)
        """
        result = RingTensor.stack([e.item for e in tensor_list], dim)
        return cls(result)

    @classmethod
    def roll(cls, input, shifts, dims=0):
        """
        Roll the elements of an BSS instance along a specified dimension.

        This method rolls the elements of the `RingTensor` in the specified `input` along
        the `dims` dimension by the number of positions specified in `shifts`.

        Args:
            input (BooleanSecretSharing): The BSS instance whose elements will be rolled.
            shifts (int): The number of positions by which the elements are shifted.
            dims (int, optional): The dimension along which to roll the elements. Defaults to 0.

        Returns:
            BooleanSecretSharing: A new BSS instance with the rolled elements.

        Examples:
            >>> res = BooleanSecretSharing.roll(bss, shifts=1)
        """
        result = RingTensor.roll(input.item, shifts, dims)
        return cls(result)

    @classmethod
    def rotate(cls, input, shifts):
        """
        Rotate each row of a 2-dimensional BSS instance to the left or right.

        This method rotates each row of the 2D `RingTensor` in the specified `input` by the
        number of positions specified in `shifts`.

        Args:
            input (BooleanSecretSharing): The 2-dimensional BSS instance to rotate.
            shifts (int): The number of positions to rotate each row. Positive values rotate to the right, negative to the left.

        Returns:
            BooleanSecretSharing: A new BSS instance with the rotated rows.

        Examples:
            >>> res = BooleanSecretSharing.rotate(bss, shifts=1)
        """
        result = RingTensor.rotate(input.item, shifts)
        return cls(result)

    @staticmethod
    def recon_from_shares(share_0, share_1):
        """
        Reconstruct the original data from two BSS instance(or we could say two shares).

        Args:
            share_0 (BooleanSecretSharing): The first share.
            share_1 (BooleanSecretSharing): The second share.

        Returns:
            RingTensor: The original data recovered from the two shares.

        Examples:
            >>> res = BooleanSecretSharing.recon_from_shares(share0, share1)
        """
        return share_0.item ^ share_1.item

    def recon(self, party=None):
        """
        Reconstruct the original data from secret shares.

        This method is used to reconstruct the original data by combining the shares
        held by different parties. It requires communication between the parties.
        (ref: Communication between the parties.)

        Returns:
            RingTensor: The reconstructed original data as a RingTensor.

        Examples:
            >>> res = bss.recon()
        """
        if party is None:
            party = PartyCtx.get()
        party.send(self)
        other = party.recv()
        return self.item ^ other.item

    @staticmethod
    def share(x, num_of_party: int):
        """
        Perform secret sharing on a RingTensor across multiple parties in a bitwise manner.

        This method splits the input `x` into secret shares in bitwise, distributing
        the shares among the specified number of parties. The returned shares
        are still RingTensors, not BSS objects.

        Note:
            The ring size of RingTensor in this method is 2, which means that the elements in the RingTensor are either
            0 or 1.

        Args:
            x (RingTensor): The RingTensor to be secret-shared.
            num_of_party (int): The number of parties to distribute the shares.

        Returns:
            List[RingTensor]: A list of RingTensor shares, one for each party.

        Examples:
            >>> shares = BooleanSecretSharing.share(ring_tensor, 2)
        """
        share_x = []
        x_0 = x.clone()

        for i in range(num_of_party - 1):
            x_i = RingTensor.random(x.shape, down_bound=0, upper_bound=2, device=DEVICE)
            share_x.append(x_i)
            x_0 ^= x_i
        share_x.append(x_0)
        return share_x
