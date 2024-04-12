"""
In this document, a super-fast (2 + 1)-PC method for Z2𝓃 is implemented by defining (2, 2)-DPFs.
Mainly refer to Storrier K, Vadapalli A, Lyons A, et al. Grotto: Screaming fast (2+ 1)-PC for ℤ2n via (2, 2)-DPFs[J].
IACR Cryptol. ePrint Arch., 2023, 2023: 108.
"""

from common.aux_parameter.parameter import Parameter
from common.tensor import *
from config.base_configs import PRG_TYPE, HALF_RING
from crypto.primitives.function_secret_sharing import *


class PPQCompareKey(Parameter):
    def __init__(self):
        self.dpf_key = DPFKey()
        self.r_in = None

    @staticmethod
    def gen(num_of_keys, beta=RingTensor(1)):
        return PPQCompare.gen(num_of_keys, beta)


class PPQCompare:
    @staticmethod
    def gen(num_of_keys: int, beta=RingTensor(1)):
        k0, k1 = PPQCompareKey(), PPQCompareKey()
        k0.r_in = RingFunc.random([num_of_keys])
        k1.r_in = RingFunc.random([num_of_keys])
        k0.dpf_key, k1.dpf_key = DPF.gen(num_of_keys, k0.r_in + k1.r_in, beta)
        return k0, k1

    @staticmethod
    def eval(x_shift: RingTensor, key, party_id, prg_type=PRG_TYPE, down_bound=RingTensor(0),
             upper_bound=RingTensor(HALF_RING - 1)):
        p = down_bound + x_shift
        q = upper_bound + x_shift
        cond = (p ^ q) < 0
        tau = ((p > q) ^ cond) * party_id
        x = RingTensor.stack([p, q]).view(2, -1, 1)

        parity_x = prefix_parity_query(x, key.dpf_key, party_id, prg_type)

        ans = (parity_x[0] ^ parity_x[1]).view(x_shift.shape) ^ tau

        return ans
