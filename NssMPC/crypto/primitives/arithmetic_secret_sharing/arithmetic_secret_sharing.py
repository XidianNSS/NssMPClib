"""Arithmetic Secret Sharing
"""
from NssMPC.common.ring import *
from NssMPC.crypto.primitives.arithmetic_secret_sharing._arithmetic_base import ArithmeticBase
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional import beaver_mul, secure_matmul, \
    secure_div, secure_eq, secure_ge, secure_exp, secure_reciprocal_sqrt, truncate, secure_tanh


class ArithmeticSecretSharing(ArithmeticBase):
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

    def __getitem__(self, item):
        return ArithmeticSecretSharing(self.item[item], self.party)

    def __setitem__(self, key, value):
        if isinstance(value, ArithmeticSecretSharing):
            self.item[key] = value.item.clone()
        else:
            raise TypeError(f"unsupported operand type(s) for setitem '{type(self)}' and {type(value)}")

    def __str__(self):
        return f"ArithmeticSecretSharing[\n{self.item}\n party:{self.party.party_id}\n]"

    def __add__(self, other):
        if isinstance(other, ArithmeticSecretSharing):
            new_tensor = self.item + other.item
            return ArithmeticSecretSharing(new_tensor, self.party)
        elif isinstance(other, RingTensor):  # for RingTensor, only party 0 add it to the share tensor
            if self.party.party_id == 0:
                new_tensor = self.item + other
            else:
                new_tensor = self.item
            return ArithmeticSecretSharing(new_tensor, self.party)
        elif isinstance(other, (int, float)):
            other = RingTensor.convert_to_ring(int(other * self.scale))
            return self + other
        else:
            raise TypeError(f"unsupported operand type(s) for + '{type(self)}' and {type(other)}")

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        if isinstance(other, ArithmeticSecretSharing):
            self.item += other.item
        elif isinstance(other, RingTensor):
            if self.party.party_id == 0:
                self.item += other
            else:
                pass
        elif isinstance(other, (int, float)):
            self += RingTensor.convert_to_ring(int(other * self.scale))
        else:
            raise TypeError(f"unsupported operand type(s) for += '{type(self)}' and {type(other)}")
        return self

    def __sub__(self, other):
        if isinstance(other, ArithmeticSecretSharing):
            new_tensor = self.item - other.item
            return ArithmeticSecretSharing(new_tensor, self.party)
        elif isinstance(other, RingTensor):
            if self.party.party_id == 0:
                new_tensor = self.item - other
            else:
                new_tensor = self.item
            return ArithmeticSecretSharing(new_tensor, self.party)
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
            if self.party.party_id == 0:
                self.item -= other
            else:
                pass
        elif isinstance(other, (int, float)):
            self -= RingTensor.convert_to_ring(int(other * self.scale))
        else:
            raise TypeError(f"unsupported operand type(s) for -= '{type(self)}' and {type(other)}")
        return self

    def __mul__(self, other):
        if isinstance(other, ArithmeticSecretSharing):
            res = beaver_mul(self, other)
        elif isinstance(other, RingTensor):
            res = ArithmeticSecretSharing(self.item * other, self.party)
        elif isinstance(other, int):
            return ArithmeticSecretSharing(self.item * other, self.party)
        else:
            raise TypeError(f"unsupported operand type(s) for * '{type(self)}' and {type(other)}")

        return res / self.scale

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        if isinstance(other, ArithmeticSecretSharing):
            res = secure_matmul(self, other)
        elif isinstance(other, RingTensor):
            res = ArithmeticSecretSharing(RingTensor.matmul(self.item, other), self.party)
        else:
            raise TypeError(f"unsupported operand type(s) for @ '{type(self)}' and {type(other)}")

        return res / self.scale

    def __pow__(self, power, modulo=None):
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
        return -self >= -other

    def __gt__(self, other):
        le = self <= other
        return -(le - 1)

    def __lt__(self, other):
        ge = self >= other
        return -(ge - 1)

    @classmethod
    def cat(cls, tensor_list, dim=0):
        result = RingTensor.cat([e.item for e in tensor_list], dim)
        return cls(result, tensor_list[0].party)

    @classmethod
    def stack(cls, tensor_list, dim=0):
        result = RingTensor.stack([e.item for e in tensor_list], dim)
        return cls(result, tensor_list[0].party)

    @classmethod
    def gather(cls, input, dim, index):
        result = RingTensor.gather(input.item, dim, index)
        return cls(result, input.party)

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

    @classmethod
    def exp(cls, x):
        return secure_exp(x)

    @classmethod
    def rsqrt(cls, x):
        return secure_reciprocal_sqrt(x)

    @classmethod
    def tanh(cls, x):
        return secure_tanh(x)

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
        return share_0.item + share_1.item

    def restore(self):
        """
        Restore the original data
        Both parties involved need to communicate

        Returns:
            RingTensor: the original data
        """
        self.party.send(self)
        other = self.party.receive()
        return self.item + other.item

    @classmethod
    def share(cls, tensor: RingTensor, num_of_party: int = 2):
        """
        Perform secret sharing via addition on a RingTensor.
        Note that the return is still a list of RingTensor, not ArithmeticRingTensor.

        Args:
            tensor: the RingTensor to be shared
            num_of_party: the number of party

        Returns:
            []: a list of RingTensor
        """
        share = []
        x_0 = tensor.clone()

        for i in range(num_of_party - 1):
            x_i = RingTensor.random(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            share.append(cls(x_i))
            x_0 -= x_i
        share.append(cls(x_0))
        return share
