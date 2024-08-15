"""
This document defines function secret sharing for distributed interval comparison functions(DICF)
in secure two-party computing.

The functions and definitions can refer to E. Boyle e.t.c. Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation.2021
https://link.springer.com/chapter/10.1007/978-3-030-77886-6_30
"""
from NssMPC.common.ring import RingTensor
from NssMPC.config.configs import PRG_TYPE, HALF_RING, data_type
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dicf_key import DICFKey, GrottoDICFKey, SigmaDICFKey
from NssMPC.crypto.primitives.function_secret_sharing.dcf import DCF
from NssMPC.crypto.primitives.function_secret_sharing.dpf import prefix_parity_query


class DICF:
    @staticmethod
    def gen(num_of_keys, down_bound=RingTensor(0), upper_bound=RingTensor(HALF_RING - 1)):
        return DICFKey.gen(num_of_keys, down_bound, upper_bound)

    @staticmethod
    def eval(x_shift, keys, party_id, down_bound=RingTensor(0), upper_bound=RingTensor(HALF_RING - 1)):
        p = down_bound
        q = upper_bound

        q1 = q + 1

        xp = (x_shift + (-1 - p))
        xq1 = (x_shift + (-1 - q1))

        s_p = DCF.eval(xp, keys.dcf_key, party_id, prg_type=PRG_TYPE)
        s_q = DCF.eval(xq1, keys.dcf_key, party_id, prg_type=PRG_TYPE)

        res = party_id * (((x_shift > p) + 0) - ((x_shift > q1) + 0)) - s_p + s_q + keys.z
        return res.to(data_type)


"""
In this document, a super-fast (2 + 1)-PC method for Z2𝓃 is implemented by defining (2, 2)-DPFs.
Mainly refer to Storrier K, Vadapalli A, Lyons A, et al. Grotto: Screaming fast (2+ 1)-PC for ℤ2n via (2, 2)-DPFs[J].
IACR Cryptol. ePrint Arch., 2023, 2023: 108.
"""


class GrottoDICF:
    @staticmethod
    def gen(num_of_keys: int, beta=RingTensor(1)):
        return GrottoDICFKey.gen(num_of_keys, beta)

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


"""
The DReLU protocol in the paper "SIGMA: Secure GPT Inference with Function Secret Sharing"
https://eprint.iacr.org/2023/1269
"""


class SigmaDICF:
    @staticmethod
    def gen(num_of_keys):
        return SigmaDICFKey.gen(num_of_keys)

    @staticmethod
    def eval(x_shift: RingTensor, key, party_id):
        shape = x_shift.shape
        x_shift = x_shift.view(-1, 1)
        y = x_shift % (HALF_RING - 1)
        y = y + 1
        out = prefix_parity_query(y, key.dpf_key, party_id)
        out = x_shift.signbit() * party_id ^ key.c.view(-1, 1) ^ out
        return out.view(shape)

    @staticmethod
    def one_key_eval(input_list, key, party_id):
        """
        eval multiple inputs with one key, can be used only when the input data is the offset of the same number
        Args:
            input_list:
            key:
            party_id:

        Returns:

        """
        num = len(input_list)
        x_shift = RingTensor.stack(input_list)
        shape = x_shift.shape
        x_shift = x_shift.view(num, -1, 1)
        y = x_shift % (HALF_RING - 1)
        y = y + 1
        out = prefix_parity_query(y, key.dpf_key, party_id)
        out = x_shift.signbit() * party_id ^ key.c.view(-1, 1) ^ out
        return out.view(shape)
