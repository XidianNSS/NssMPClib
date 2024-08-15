"""
Replicated Secret Sharing
"""
import torch

from NssMPC.common.ring import *
from NssMPC.crypto.primitives.arithmetic_secret_sharing._arithmetic_base import ArithmeticBase, RingPair
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import v_mul, v_matmul, truncate
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.comparison import \
    secure_ge as v_secure_ge
from NssMPC.crypto.protocols.replicated_secret_sharing.semi_honest_functional import (secure_ge, mul_with_out_trunc,
                                                                                      matmul_with_out_trunc)
from NssMPC.secure_model.mpc_party import HonestMajorityParty


class ReplicatedSecretSharing(ArithmeticBase):
    def __init__(self, ring_pair, party=None):
        if isinstance(ring_pair, list):
            ring_pair = RingPair(ring_pair[0], ring_pair[1])
        super(ReplicatedSecretSharing, self).__init__(ring_pair, party)

    @property
    def T(self):
        return self.__class__(self.item.T, self.party)

    def __getitem__(self, item):
        return ReplicatedSecretSharing([self.item[0][item], self.item[1][item]], self.party)

    def __setitem__(self, key, value):
        if isinstance(value, ReplicatedSecretSharing):
            self.item[0][key] = value.item[0].clone()
            self.item[1][key] = value.item[1].clone()
        else:
            raise TypeError(f"unsupported operand type(s) for setitem '{type(self)}' and {type(value)}")

    def __str__(self):
        return f"[{self.__class__.__name__}\n {self.item}]"

    def __add__(self, other):
        if isinstance(other, ReplicatedSecretSharing):
            return self.__class__(RingPair(self.item[0] + other.item[0], self.item[1] + other.item[1]), self.party)
        elif isinstance(other, RingTensor):
            zeros = RingTensor.zeros_like(other, dtype=other.dtype, device=other.device)
            if self.party.party_id == 0:
                return self.__class__(RingPair(self.item[0] + other, self.item[1] + zeros), self.party)
            elif self.party.party_id == 2:
                return self.__class__(RingPair(self.item[0] + zeros, self.item[1] + other), self.party)
            else:
                return self.__class__(RingPair(self.item[0] + zeros, self.item[1] + zeros), self.party)
        elif isinstance(other, int):
            if self.party.party_id == 0:
                return self.__class__(RingPair(self.item[0] + other * self.scale, self.item[1]), self.party)
            elif self.party.party_id == 2:
                return self.__class__(RingPair(self.item[0], self.item[1] + other * self.scale), self.party)
            else:
                return self.clone()

        else:
            TypeError("unsupported operand type(s) for + 'ReplicatedSecretSharing' and ", type(other))

    def __sub__(self, other):
        if isinstance(other, ReplicatedSecretSharing):
            return self.__class__(RingPair(self.item[0] - other.item[0], self.item[1] - other.item[1]), self.party)
        elif isinstance(other, RingTensor):
            zeros = RingTensor.zeros_like(other, dtype=other.dtype, device=other.device)
            if self.party.party_id == 0:
                return self.__class__(RingPair(self.item[0] - other, self.item[1] - zeros), self.party)
            elif self.party.party_id == 2:
                return self.__class__(RingPair(self.item[0] - zeros, self.item[1] - other), self.party)
            else:
                return self.__class__(RingPair(self.item[0] - zeros, self.item[1] - zeros), self.party)
        elif isinstance(other, int):
            if self.party.party_id == 0:
                return self.__class__(RingPair(self.item[0] - other * self.scale, self.item[1]), self.party)
            elif self.party.party_id == 2:
                return self.__class__(RingPair(self.item[0], self.item[1] - other * self.scale), self.party)
            else:
                return self.clone()
        else:
            TypeError("unsupported operand type(s) for - 'ReplicatedSecretSharing' and ", type(other))

    def __mul__(self, other):
        if isinstance(other, ReplicatedSecretSharing):
            if isinstance(self.party, HonestMajorityParty):
                return v_mul(self, other)
            else:
                result = mul_with_out_trunc(self, other)
                if self.item[0].dtype == "float":
                    result = truncate(result)
                return result
        elif isinstance(other, RingTensor) or isinstance(other, int):
            result = self.item * other
            return ReplicatedSecretSharing(result, self.party)
        else:
            TypeError("unsupported operand type(s) for * 'ReplicatedSecretSharing' and ", type(other))

    def __matmul__(self, other):
        if isinstance(other, ReplicatedSecretSharing):
            if isinstance(self.party, HonestMajorityParty):
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
            result0 = self.item[0] @ other
            result1 = self.item[1] @ other
            torch.cuda.empty_cache()
            return ReplicatedSecretSharing(RingPair(result0, result1), self.party)
        else:
            TypeError("unsupported operand type(s) for @ 'ReplicatedSecretSharing' and ", type(other))

    def __ge__(self, other):
        if isinstance(self.party, HonestMajorityParty):
            ge = v_secure_ge
        else:
            ge = secure_ge
        if isinstance(other, ReplicatedSecretSharing):
            return ge(self, other)
        elif isinstance(other, RingTensor):
            zeros = RingTensor.zeros_like(other, dtype=other.dtype, device=other.device)
            if self.party.party_id == 0:
                other = ReplicatedSecretSharing([other, zeros], self.party)
            elif self.party.party_id == 1:
                other = ReplicatedSecretSharing([zeros, zeros], self.party)
            elif self.party.party_id == 2:
                other = ReplicatedSecretSharing([zeros, other], self.party)
            return ge(self, other)
        elif isinstance(other, int):
            return self >= RingTensor.convert_to_ring(int(other * self.scale))
        elif isinstance(other, float):
            assert self.dtype == 'float', "only float can compare with float"
            return self >= RingTensor.convert_to_ring(other)
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
        result_0 = RingTensor.cat([e.item[0] for e in tensor_list], dim)
        result_1 = RingTensor.cat([e.item[1] for e in tensor_list], dim)
        return cls(RingPair(result_0, result_1), tensor_list[0].party)

    @classmethod
    def stack(cls, tensor_list, dim=0):
        result_0 = RingTensor.stack([e.item[0] for e in tensor_list], dim)
        result_1 = RingTensor.stack([e.item[1] for e in tensor_list], dim)
        return cls(RingPair(result_0, result_1), tensor_list[0].party)

    @classmethod
    def roll(cls, input, shifts, dims=0):
        result_0 = RingTensor.roll(input.item[0], shifts, dims)
        result_1 = RingTensor.roll(input.item[1], shifts, dims)
        return cls(RingPair(result_0, result_1), input.party)

    @classmethod
    def rotate(cls, input, shifts):
        """
        For the input 2-dimensional ArithmeticRingTensor, rotate each row to the left or right
        """
        result_0 = RingTensor.rotate(input.item[0], shifts)
        result_1 = RingTensor.rotate(input.item[1], shifts)
        return cls(RingPair(result_0, result_1), input.party)

    @staticmethod
    def gen_and_share(r_tensor, party):
        r0, r1, r2 = ReplicatedSecretSharing.share(r_tensor)
        r0.party = party
        r1.party = party
        r2.party = party
        party.send((party.party_id + 1) % 3, r1)
        party.send((party.party_id + 2) % 3, r2)
        return r0

    @classmethod
    def share(cls, tensor: RingTensor):
        """
        对输入RingTensor进行三方复制秘密分享。

        :param tensor: 进行秘密分享的输入数据张量,类型为RingTensor。
        :return: 复制秘密分享后的分享份额列表，包含三个RingTensor的二元列表。
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
        基于三方复制秘密分享的数据张量的明文值恢复。
        :return: 恢复后的数据张量，类型为RingTensor。
        """
        # 发送部分
        self.party.send((self.party.party_id + 1) % 3, self.item[0])
        # 接收部分
        other = self.party.receive((self.party.party_id + 2) % 3)
        return self.item[0] + self.item[1] + other

    @staticmethod
    def reshare(value: RingTensor, party):
        r_0 = party.prg_0.random(value.numel())
        r_1 = party.prg_1.random(value.numel())
        r_0 = r_0.reshape(value.shape)
        r_1 = r_1.reshape(value.shape)
        r_0.dtype = r_1.dtype = value.dtype
        value = value.tensor + r_0 - r_1
        party.send((party.party_id + 2) % 3, value)
        other = party.receive((party.party_id + 1) % 3)
        return ReplicatedSecretSharing(RingPair(value, other), party)

    @classmethod
    def random(cls, shape, party):
        num_of_value = 1
        for d in shape:
            num_of_value *= d
        r_0 = party.prg_0.random(num_of_value)
        r_1 = party.prg_1.random(num_of_value)
        r = ReplicatedSecretSharing([r_0, r_1], party)
        r = r.reshape(shape)
        return r

    @classmethod
    def rand_like(cls, x, party):
        r = ReplicatedSecretSharing.random(x.shape, party)
        if isinstance(x, RingTensor):
            r.dtype = x.dtype
        return r

    @classmethod
    def empty(cls, size, dtype='int', party=None):
        r = RingTensor.empty(size, dtype)
        return cls(r, party)

    @classmethod
    def empty_like(cls, tensor, party=None):
        r = RingTensor.empty_like(tensor)
        r.party = party if party else tensor.party
        return r

    @classmethod
    def zeros(cls, shape, dtype='int', party=None):
        r = RingTensor.zeros(shape, dtype)
        return cls(r, party)

    @classmethod
    def zeros_like(cls, tensor, dtype='int', party=None):
        r = RingTensor.zeros_like(tensor, dtype)
        r.party = party if party else tensor.party
        return r
