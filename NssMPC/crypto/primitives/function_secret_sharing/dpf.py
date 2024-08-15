"""
This document defines function secret sharing for distributed point functions(DPF) in secure two-party computing.
The process of generating and evaluating distributed point function keys involves relevant procedures of DPF.
The functions and definitions can refer to E. Boyle e.t.c. Function Secret Sharing: Improvements and Extensions.2016
https://dl.acm.org/doi/10.1145/2976749.2978429
"""
from NssMPC.common.random.prg import PRG
from NssMPC.common.ring import RingTensor
from NssMPC.common.utils import convert_tensor
from NssMPC.config.configs import LAMBDA, DEVICE, PRG_TYPE
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.cw import CW
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dpf_key import DPFKey


class DPF:
    @staticmethod
    def gen(num_of_keys: int, alpha: RingTensor, beta):
        return DPFKey.gen(num_of_keys, alpha, beta)

    @staticmethod
    def eval(x: RingTensor, keys: DPFKey, party_id: int, prg_type=PRG_TYPE):
        """
        the api of dpf EVAL progress
        Using input x, the party compute the function value locally, which is the shared value of the original function.

        Args:
            x: the input
            keys: the key of the secret function sharing
            party_id: the id of the party
            prg_type: the type of pseudorandom number generator

        Returns:
            the value of the DPF
        """
        x = x.clone()

        prg = PRG(prg_type, DEVICE)

        t_last = party_id

        s_last = keys.s

        for i in range(x.bit_len):
            cw = keys.cw_list[i]

            s_cw = cw.s_cw
            t_cw_l = cw.t_cw_l
            t_cw_r = cw.t_cw_r

            s_l, t_l, s_r, t_r = CW.gen_dpf_cw(prg, s_last, LAMBDA)

            s1_l = s_l ^ (s_cw * t_last)
            t1_l = t_l ^ (t_cw_l * t_last)
            s1_r = s_r ^ (s_cw * t_last)
            t1_r = t_r ^ (t_cw_r * t_last)

            x_shift_bit = x.get_tensor_bit(x.bit_len - 1 - i)

            s_last = s1_r * x_shift_bit + s1_l * (1 - x_shift_bit)
            t_last = t1_r * x_shift_bit + t1_l * (1 - x_shift_bit)

        dpf_result = pow(-1, party_id) * (convert_tensor(s_last) + t_last * keys.ex_cw_dpf)

        return RingTensor(dpf_result, x.dtype, x.device)


def prefix_parity_query(x: RingTensor, keys, party_id, prg_type=PRG_TYPE):
    """
    By transforming the distributed point function EVAL process
    to compute the prefix parity sum of a section (Prefix Parity Sum)

    Based on the input x, the participant locally computes the parity of the point in the construction tree

    Args:
        x: the input
        keys: the key of distributed point function
        party_id: the id of the party
        prg_type: the type of pseudorandom number generator

    Returns:
        the result of the prefix parity sum
    """
    prg = PRG(prg_type, DEVICE)

    d = 0
    psg_b = 0
    t_last = party_id

    s_last = keys.s
    for i in range(x.bit_len):
        cw = keys.cw_list[i]

        s_cw = cw.s_cw
        t_cw_l = cw.t_cw_l
        t_cw_r = cw.t_cw_r

        s_l, t_l, s_r, t_r = CW.gen_dpf_cw(prg, s_last, LAMBDA)

        s1_l = s_l ^ (s_cw * t_last)
        t1_l = t_l ^ (t_cw_l * t_last)
        s1_r = s_r ^ (s_cw * t_last)
        t1_r = t_r ^ (t_cw_r * t_last)

        x_shift_bit = x.get_tensor_bit(x.bit_len - 1 - i)

        cond = (d != x_shift_bit)
        d = x_shift_bit * cond + d * ~cond

        psg_b = (psg_b ^ t_last) * cond + psg_b * ~cond

        s_last = s1_r * x_shift_bit + s1_l * (1 - x_shift_bit)
        t_last = t1_r * x_shift_bit + t1_l * (1 - x_shift_bit)

    psg_b = (psg_b ^ t_last) * d + psg_b * (1 - d)

    return RingTensor(psg_b, x.dtype, x.device)
