"""Boolean Secret Sharing
"""
from NssMPC.common.ring import RingTensor
from NssMPC.crypto.protocols.boolean_secret_sharing.semi_honest_functional.semi_honest_functional import beaver_and
from NssMPC.common.ring import *
from NssMPC.crypto.primitives.arithmetic_secret_sharing._arithmetic_base import ArithmeticBase


class BooleanSecretSharing(ArithmeticBase):
    """
    The RingTensor that supports arithmetic secret sharing

    Attributes:
        party: the party that holds the ArithmeticSharedRingTensor
    """

    def __init__(self, ring_tensor, party=None):
        assert isinstance(ring_tensor, RingTensor), "ring_tensor must be a RingTensor"
        super().__init__(ring_tensor, party)

    @property
    def ring_tensor(self):
        return self.item

    @property
    def T(self):
        return self.__class__(self.item.T, self.party)

    def __str__(self):
        return f"ArithmeticSecretSharing[\n{self.item}\n party:{self.party.party_id}\n]"

    def __invert__(self):
        """
        Args:
            other: ArithmeticSharedRingTensor or RingTensor
        """
        if self.party.party_id == 0:
            new_tensor = ~self.item
            return BooleanSecretSharing(new_tensor, self.party)
        else:
            return BooleanSecretSharing(self.item, self.party)

    def __and__(self, other):
        """
        Args:
            other: ArithmeticSharedRingTensor or RingTensor
        """
        if isinstance(other, BooleanSecretSharing):
            new_tensor = beaver_and(self, other)
            return BooleanSecretSharing(new_tensor, self.party)
        elif isinstance(other, RingTensor):
            return BooleanSecretSharing(self.item & other, self.party)
        else:
            raise TypeError(f"unsupported operand type(s) for + '{type(self)}' and {type(other)}")

    @classmethod
    def cat(cls, tensor_list, dim=0):
        result = RingTensor.cat([e.item for e in tensor_list], dim)
        return cls(result, tensor_list[0].party)

    @classmethod
    def stack(cls, tensor_list, dim=0):
        result = RingTensor.stack([e.item for e in tensor_list], dim)
        return cls(result, tensor_list[0].party)

    @classmethod
    def roll(cls, input, shifts, dims=0):
        result = RingTensor.roll(input.item, shifts, dims)
        return cls(result, input.party)

    @classmethod
    def rotate(cls, input, shifts):
        """
        For the input 2-dimensional ArithmeticRingTensor, rotate each row to the left or right
        """
        result = RingTensor.rotate(input.item, shifts)
        return cls(result, input.party)

    @staticmethod
    def restore_from_shares(share_0, share_1):
        """
        Recover the original data with two shared values

        Args:
            share_0: ArithmeticSharedRingTensor
            share_1: ArithmeticSharedRingTensor

        Returns:
            RingTensor: the original data
        """
        return share_0.item ^ share_1.item

    def restore(self):
        """
        Restore the original data
        Both parties involved need to communicate

        Returns:
            RingTensor: the original data
        """
        self.party.send(self)
        other = self.party.receive()
        return self.item ^ other.item

    @staticmethod
    def share(x, bit_len, num_of_party: int):
        """
        Secretly share the binary string x

        Args:
            x (tensor): the tensor to be shared
            bit_len (int): the length of the binary string
            num_of_party (int): the number of parties

        Returns:
            the list of secret sharing values
        """

        share_x = []
        x_0 = x.clone()

        for i in range(num_of_party - 1):
            x_i = RingTensor.random([bit_len], down_bound=0, upper_bound=2, device='cpu')
            share_x.append(x_i)
            x_0 ^= x_i
        share_x.append(x_0)
        return share_x
