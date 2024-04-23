"""Arithmetic Secret Sharing
"""
from common.tensor import *
from config.base_configs import DTYPE
from crypto.protocols.arithmetic_secret_sharing import *


def override_with_party(method_names):
    def decorator(cls):
        def override(method_name):
            def delegate(self, *args, **kwargs):
                result = getattr(super(cls, self), method_name)(*args, **kwargs)
                result.party = self.party
                return result

            return delegate

        for name in method_names:
            setattr(cls, name, override(name))
        return cls

    return decorator


@override_with_party(
    ['__getitem__', '__neg__', 'reshape', 'view', 'transpose', 'squeeze', 'unsqueeze', 'flatten', 'clone',
     'pad', 'sum', 'repeat', 'permute'])
class ArithmeticSharedRingTensor(RingTensor):
    """
    The RingTensor that supports arithmetic secret sharing

    Attributes:
        party: the party that holds the ArithmeticSharedRingTensor
    """

    @singledispatchmethod
    def __init__(self):
        super().__init__(self)
        self.party = None

    @__init__.register(torch.Tensor)
    def _from_tensor(self, tensor, party=None, dtype=DTYPE, device=DEVICE):
        assert tensor.dtype in [torch.int64, torch.int32], f"unsupported data type {tensor.dtype}"
        super(ArithmeticSharedRingTensor, self).__init__(tensor, dtype, device)
        self.party = party

    @__init__.register(RingTensor)
    def _from_ring_tensor(self, tensor: RingTensor, party=None):
        super(ArithmeticSharedRingTensor, self).__init__(tensor.tensor, tensor.dtype, tensor.device)
        self.party = party

    @property
    def ring_tensor(self):
        return RingTensor(self.tensor, dtype=self.dtype, device=self.device)

    @property
    def T(self):
        result = super().T
        result.party = self.party
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state['party'] = None
        return state

    def __str__(self):
        return "{}\n party:{}".format(super(ArithmeticSharedRingTensor, self).__str__(), self.party.party_id)

    def __add__(self, other):
        """
        Args:
            other: ArithmeticSharedRingTensor or RingTensor
        """
        if isinstance(other, ArithmeticSharedRingTensor):
            new_tensor = self.tensor + other.tensor
            return ArithmeticSharedRingTensor(new_tensor, self.party, self.dtype, self.device)
        elif isinstance(other, RingTensor):  # for RingTensor, only party 0 add it to the share tensor
            if self.party.party_id == 0:
                new_tensor = self.tensor + other.tensor
            else:
                new_tensor = self.tensor
            return ArithmeticSharedRingTensor(new_tensor, self.party, self.dtype, self.device)
        elif isinstance(other, (int, float)):
            other = RingTensor.convert_to_ring(int(other * self.scale))
            return self + other
        else:
            raise TypeError(f"unsupported operand type(s) for + '{type(self)}' and {type(other)}")

    def __sub__(self, other):
        """
        Args:
            other: ArithmeticSharedRingTensor or RingTensor
        """
        if isinstance(other, ArithmeticSharedRingTensor):
            new_tensor = self.tensor - other.tensor
            return ArithmeticSharedRingTensor(new_tensor, self.party, self.dtype, self.device)
        elif isinstance(other, RingTensor):
            if self.party.party_id == 0:
                new_tensor = self.tensor - other.tensor
            else:
                new_tensor = self.tensor
            return ArithmeticSharedRingTensor(new_tensor, self.party, self.dtype, self.device)
        elif isinstance(other, (int, float)):
            other = RingTensor.convert_to_ring(int(other * self.scale))
            return self - other
        else:
            raise TypeError(f"unsupported operand type(s) for - '{type(self)}' and {type(other)}")

    def __mul__(self, other):
        if isinstance(other, ArithmeticSharedRingTensor):
            res = beaver_mul(self, other)
        elif isinstance(other, RingTensor):
            res = ArithmeticSharedRingTensor(self.tensor * other.tensor, self.party, self.dtype, self.device)
        elif isinstance(other, int):
            return ArithmeticSharedRingTensor(self.tensor * other, self.party, self.dtype, self.device)
        else:
            raise TypeError(f"unsupported operand type(s) for * '{type(self)}' and {type(other)}")

        return res / self.scale

    def __matmul__(self, other):
        if isinstance(other, ArithmeticSharedRingTensor):
            res = secure_matmul(self, other)
        elif isinstance(other, RingTensor):
            res = ArithmeticSharedRingTensor(cuda_matmul(self.tensor, other.tensor), self.party, self.dtype,
                                             self.device)
        else:
            raise TypeError(f"unsupported operand type(s) for @ '{type(self)}' and {type(other)}")

        return res / self.scale

    def __pow__(self, power, modulo=None):
        # TODO: 'continue' coming soon. fast power?
        if isinstance(power, int):
            temp = self
            res = temp
            for i in range(1, power):
                res = res * temp
            return res
        else:
            raise TypeError(f"unsupported operand type(s) for ** '{type(self)}' and {type(power)}")

    def __truediv__(self, other):
        if other == 1:
            return self
        # TODO: support dtype = 'int'
        assert self.dtype == 'float', 'only support float type for division'
        if isinstance(other, int):
            temp = self.clone()
            return truncate(temp, other)
        elif isinstance(other, float):
            return (self * self.scale) / int(other * self.scale)
        elif isinstance(other, ArithmeticSharedRingTensor):
            return secure_div(self, other)
        elif isinstance(other, RingTensor):
            return truncate(self * other.scale, other.tensor)
        else:
            raise TypeError(f"unsupported operand type(s) for / '{type(self)}' and {type(other)}")

    def __eq__(self, other):
        if isinstance(other, RingTensor):
            return secure_eq(self, other)
        elif isinstance(other, int):
            return secure_eq(self, RingTensor.convert_to_ring(int(other * self.scale)))
        elif isinstance(other, float):
            assert self.dtype == 'float', "only float can compare with float"
            return secure_eq(self, RingTensor.convert_to_ring(other))
        else:
            raise TypeError(f"unsupported operand type(s) for comparison '{type(self)}' and {type(other)}")

    def __ge__(self, other):
        if isinstance(other, RingTensor):
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
    def cat(cls, tensor_list, dim=0, party=None):
        result = super().cat(tensor_list)
        if party is None:
            party = tensor_list[0].party
        result.party = party
        return result

    @classmethod
    def load_from_file(cls, file_path, party=None):
        """
        Load a ArithmeticSharedRingTensor from a file
        """
        result = super().load_from_file(file_path)
        result.party = party
        return result

    @classmethod
    def roll(cls, input, shifts, dims=0, party=None):
        result = super().roll(input, shifts, dims)
        result.party = party
        return result

    @classmethod
    def rotate(cls, input, shifts, party=None):
        """
        For the input 2-dimensional ArithmeticRingTensor, rotate each row to the left or right
        """
        result = super().rotate(input, shifts)
        result.party = party
        return result

    @classmethod
    def max(cls, x, dim=None):
        def max_(inputs):
            if inputs.shape[0] == 1:
                return inputs
            if inputs.shape[0] % 2 == 1:
                inputs_ = inputs[-1:]
                inputs = ArithmeticSharedRingTensor.cat([inputs, inputs_], 0)
            inputs_0 = inputs[0::2]
            inputs_1 = inputs[1::2]
            ge = inputs_0 >= inputs_1

            return ge * (inputs_0 - inputs_1) + inputs_1

        if dim is None:
            x = x.flatten()
        else:
            x = x.transpose(dim, 0)
        if x.shape[0] == 1:
            return x.transpose(dim, 0).squeeze(-1)
        else:
            x = max_(x)
        return ArithmeticSharedRingTensor.max(x.transpose(0, dim), dim)

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
        return share_0.ring_tensor + share_1.ring_tensor

    def restore(self):
        """
        Restore the original data
        Both parties involved need to communicate

        Returns:
            RingTensor: the original data
        """
        self.party.send(self)
        other = self.party.receive()
        return self.ring_tensor + other.ring_tensor

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
            x_i = RingFunc.random(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            share.append(cls(x_i))
            x_0 -= x_i
        share.append(cls(x_0))
        return share
