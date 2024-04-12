"""
This document defines function secret sharing for distributed interval comparison functions(DICF)
in secure two-party computing.

The functions and definitions can refer to E. Boyle e.t.c. Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation.2021
https://link.springer.com/chapter/10.1007/978-3-030-77886-6_30
"""

from common.aux_parameter.parameter import Parameter
from common.tensor import *
from config.base_configs import PRG_TYPE, HALF_RING, data_type
from crypto.primitives.function_secret_sharing import *


class DICFKey(Parameter):
    """
    The secret sharing key for distributed interval comparison function

    The generation of the DICF key is based on the DCF key

    Attributes:
        dcf_key: the key of DICF
        r_in: the offset of function
        z: the check bit
    """

    def __init__(self):
        self.dcf_key = DCFKey()
        self.r_in = None
        self.z = None
        self.size = 0

    @staticmethod
    def gen(num_of_keys, down_bound=RingTensor(0), upper_bound=RingTensor(HALF_RING - 1)):
        """
        The api generating DICF key, which can generate multiple keys.

        Args:
            num_of_keys: the number of DICF key
            down_bound: the down bound of the DICF, default is 0
            upper_bound: the upper bound of the DICF, default is HALF_RING-1

        Returns:
            the participants' keyes
        """
        return DICF.gen(num_of_keys, down_bound, upper_bound)


class DICF:
    @staticmethod
    def gen(num_of_keys, down_bound=RingTensor(0), upper_bound=RingTensor(HALF_RING - 1)):
        r_in = RingFunc.random([num_of_keys, 1])
        n_sub_1 = RingTensor(-1)

        gamma = r_in + n_sub_1

        b = RingTensor(1)

        # 修正参数
        q1 = (upper_bound + 1)
        ap = (down_bound + r_in)
        aq = (upper_bound + r_in)
        aq1 = (upper_bound + 1 + r_in)

        out = ((ap > aq) + 0) - ((ap > down_bound) + 0) + ((aq1 > q1) + 0) + (
                (aq == n_sub_1) + 0)

        k0 = DICFKey()
        k1 = DICFKey()

        keys = DCF.gen(num_of_keys, gamma, b)

        k0.dcf_key, k1.dcf_key = keys

        z_share = RingFunc.random([num_of_keys])
        r_share = RingFunc.random([num_of_keys])

        k0.z, k1.z = out.squeeze(1) - z_share, z_share
        k0.r_in, k1.r_in = r_in.squeeze(1) - r_share, r_share

        return k0, k1

    @staticmethod
    def eval(x_shift, keys, party_id, down_bound=RingTensor(0), upper_bound=RingTensor(HALF_RING - 1)):
        p = down_bound
        q = upper_bound
        n_1 = RingTensor(-1)

        q1 = q + 1

        xp = (x_shift + (n_1 - p))
        xq1 = (x_shift + (n_1 - q1))

        s_p = DCF.eval(xp, keys.dcf_key, party_id, prg_type=PRG_TYPE)
        s_q = DCF.eval(xq1, keys.dcf_key, party_id, prg_type=PRG_TYPE)

        res = party_id * (((x_shift > p) + 0) - ((x_shift > q1) + 0)) - s_p + s_q + keys.z

        return res.to(data_type)
