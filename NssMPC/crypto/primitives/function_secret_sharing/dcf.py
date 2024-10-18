"""

This document defines function secret sharing for distributed comparison functions(DCF) in secure two-party computing.
The implementation is based on the work of E. Boyle e.t.c. Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation.2021
For reference, see the `paper <https://link.springer.com/chapter/10.1007/978-3-030-77886-6_30>`_.
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC import RingTensor
from NssMPC.common.random.prg import PRG
from NssMPC.common.utils import convert_tensor
from NssMPC.config.configs import LAMBDA, DEVICE, PRG_TYPE
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.cw import CW
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dcf_key import DCFKey


class DCF:
    """
    **FSS for Distributed Comparison Function (DCF)**.

    This class implements the generation of keys and the evaluation for computing DCF securely using function secret sharing.
    """

    @staticmethod
    def gen(num_of_keys, alpha, beta):
        """
        Generate DCF keys.

        This method generates the DCF keys required for secure comparison.

        .. note::
            Distributed comparison function:
                f(x)=beta, if x < alpha; f(x)=0, else


        :param num_of_keys: The number of keys to generate.
        :type num_of_keys: int
        :param alpha: The comparison point (private value for comparison).
        :type alpha: RingTensor
        :param beta: The output value if the comparison is true.
        :type beta: RingTensor
        :return: A tuple containing two DCFKey objects for the two parties.
        :rtype: Tuple[DCFKey, DCFKey]
        """
        return DCFKey.gen(num_of_keys, alpha, beta)

    @staticmethod
    def eval(x, keys, party_id, prg_type=PRG_TYPE):
        """
        Evaluate the DCF on input `x`.

        With this method, the party evaluates the function value locally using the input *x*,
        provided keys and party ID. It performs bitwise operations to securely
        compute the comparison result, which is the **shared value** of the result of the original function.

        :param x: The input RingTensor on which the DCF is evaluated.
        :type x: RingTensor
        :param keys: The secret sharing keys required for evaluation.
        :type keys: DCFKey
        :param party_id: The party ID (0 or 1), identifying which party is performing the evaluation.
        :type party_id: int
        :param prg_type: The type of pseudorandom generator (PRG) used during evaluation, defaults to PRG_TYPE.
        :type prg_type: str, optional
        :return: The sharing value of the comparison result for each party as a RIngTensor.
        :rtype: RingTensor
        """
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
