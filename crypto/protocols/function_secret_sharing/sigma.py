"""
The DReLU protocol in the paper "SIGMA: Secure GPT Inference with Function Secret Sharing"
https://eprint.iacr.org/2023/1269
"""

from common.aux_parameter.parameter import Parameter
from common.tensor import *
from config.base_configs import HALF_RING
from crypto.primitives.function_secret_sharing import *


class SigmaCompareKey(Parameter):
    def __init__(self):
        self.dpf_key = DPFKey()
        self.c = None
        self.r_in = None
        self.size = 0

    @staticmethod
    def gen(num_of_keys):
        return SigmaCompare.gen(num_of_keys)


class SigmaCompare:
    @staticmethod
    def gen(num_of_keys):
        k0 = SigmaCompareKey()
        k1 = SigmaCompareKey()

        k0.r_in = RingFunc.random([num_of_keys])
        k1.r_in = RingFunc.random([num_of_keys])
        r_in = k0.r_in + k1.r_in

        y1 = r_in % (HALF_RING - 1)
        k0.dpf_key, k1.dpf_key = DPF.gen(num_of_keys, y1, RingTensor(1))
        c = r_in.signbit()
        c0 = RingFunc.random([num_of_keys], down_bound=0, upper_bound=2)
        c1 = c ^ c0

        k0.c = c0
        k1.c = c1

        return k0, k1

    @staticmethod
    def eval(x_shift: RingTensor, key, party_id):
        shape = x_shift.shape
        x_shift = x_shift.view(-1, 1)
        y = x_shift % (HALF_RING - 1)
        y = y + 1
        out = prefix_parity_query(y, key.dpf_key, party_id)
        out = x_shift.signbit() * party_id ^ key.c.view(-1, 1) ^ out
        return out.view(shape)
