"""
This document defines function secret sharing for distributed comparison functions(DCF) in secure two-party computing.
The functions and definitions can refer to E. Boyle e.t.c. Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation.2021
https://link.springer.com/chapter/10.1007/978-3-030-77886-6_30
"""
from NssMPC import RingTensor
from NssMPC.common.random.prg import PRG
from NssMPC.common.utils import convert_tensor
from NssMPC.config.configs import LAMBDA, DEVICE, PRG_TYPE
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.cw import CW
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dcf_key import DCFKey


class DCF:
    @staticmethod
    def gen(num_of_keys, alpha, beta):
        return DCFKey.gen(num_of_keys, alpha, beta)

    @staticmethod
    def eval(x, keys, party_id, prg_type=PRG_TYPE):
        shape = x.shape
        x = x.clone()
        x = x.view(-1, 1)

        prg = PRG(prg_type, DEVICE)
        t_last = party_id
        dcf_result = 0
        s_last = keys.s

        for i in range(x.bit_len):
            cw = keys.cw_list[i]

            s_cw = cw.s_cw
            v_cw = cw.v_cw
            t_cw_l = cw.t_cw_l
            t_cw_r = cw.t_cw_r

            s_l, v_l, t_l, s_r, v_r, t_r = CW.gen_dcf_cw(prg, s_last, LAMBDA)

            s1_l = s_l ^ (s_cw * t_last)
            t1_l = t_l ^ (t_cw_l * t_last)
            s1_r = s_r ^ (s_cw * t_last)
            t1_r = t_r ^ (t_cw_r * t_last)

            x_shift_bit = x.get_tensor_bit(x.bit_len - 1 - i)

            v_curr = v_r * x_shift_bit + v_l * (1 - x_shift_bit)
            dcf_result = dcf_result + pow(-1, party_id) * (convert_tensor(v_curr) + t_last * v_cw)

            s_last = s1_r * x_shift_bit + s1_l * (1 - x_shift_bit)
            t_last = t1_r * x_shift_bit + t1_l * (1 - x_shift_bit)

        dcf_result = dcf_result + pow(-1, party_id) * (
                convert_tensor(s_last) + t_last * keys.ex_cw_dcf)

        return RingTensor(dcf_result.view(shape), x.dtype, x.device)
