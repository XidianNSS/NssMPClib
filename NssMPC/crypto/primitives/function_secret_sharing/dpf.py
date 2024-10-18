"""

This document defines function secret sharing for distributed point functions(DPF) in secure two-party computing.
The process of generating and evaluating distributed point function keys involves relevant procedures of DPF.
The implementation is based on the work of E. Boyle e.t.c. Function Secret Sharing: Improvements and Extensions.2016
For reference, see the `paper <https://dl.acm.org/doi/10.1145/2976749.2978429>`_.
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.common.random.prg import PRG
from NssMPC.common.ring import RingTensor
from NssMPC.common.utils import convert_tensor
from NssMPC.config.configs import LAMBDA, DEVICE, PRG_TYPE
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.cw import CW
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dpf_key import DPFKey


class DPF:
    """
    **FSS for Distributed Point Function (DPF)**.

    This class implements the generation and evaluation of keys for computing DPF securely using function secret sharing.
    """

    @staticmethod
    def gen(num_of_keys: int, alpha: RingTensor, beta):
        """
        Generate DPF keys.

        This method generates the DPF keys required for secure comparison.

        .. note::
            Distributed point function:
                f(x)=beta, if x = alpha; f(x)=0, else


        :param num_of_keys: The number of keys to generate.
        :type num_of_keys: int
        :param alpha: The comparison point (private value for comparison).
        :type alpha: RingTensor
        :param beta: The output value if the comparison is true.
        :type beta: RingTensor
        :return: A tuple containing two DPFKey objects for the two parties.
        :rtype: Tuple[DPFKey, DPFKey]
        """
        return DPFKey.gen(num_of_keys, alpha, beta)

    @staticmethod
    def eval(x: RingTensor, keys: DPFKey, party_id: int, prg_type=PRG_TYPE):
        """
        Evaluate the DPF on input `x`.

        With this method, the party evaluates the function value locally using the input *x*,
        provided keys and party ID. It performs bitwise operations to securely
        compute the comparison result, which is the **shared value** of the original result.

        :param x: The input RingTensor on which the DPF is evaluated.
        :type x: RingTensor
        :param keys: The secret sharing keys required for evaluation.
        :type keys: DPFKey
        :param party_id: The party ID (0 or 1), identifying which party is performing the evaluation.
        :type party_id: int
        :param prg_type: The type of pseudorandom generator (PRG) used during evaluation, defaults to **PRG_TYPE**.
        :type prg_type: str, optional
        :return: The sharing value of the comparison result for each party as a RIngTensor.
        :rtype: RingTensor
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
    Return the prefix parity query of the input *x*, thus improve the efficiency of the evaluation process.

    By transforming the distributed point function evaluation(EVAL) process
    to computing the prefix parity sum of a section (Prefix Parity Sum), we can improve the computational efficiency.
    So based on the input *x*, the participant locally computes the parity of the point in the Parity-segment tree,
    and the result will be 0 if **x < alpha**, and 1 **otherwise**.

    .. important::
        We put this method in ``dpf`` document because the key used in this method is DPFKey, but actually it
        implements the functionality of DCF.

    .. note::
        The implementation is based on the work of
        **Storrier, K., Vadapalli, A., Lyons, A., & Henry, R. (2023). Grotto: Screaming Fast (2 + 1)-PC for Z2ⁿ via (2, 2)-DPFs**.
        For reference, see the `paper <https://eprint.iacr.org/2023/108>`_.

    :param x: The input RingTensor on which the prefix parity query is performed.
    :type x: RingTensor
    :param keys: The secret sharing keys required for prefix parity query.
    :type keys: DPFKey
    :param party_id: The party ID (0 or 1), identifying which party is performing the evaluation.
    :type party_id: int
    :param prg_type: The type of pseudorandom generator (PRG) used during evaluation, defaults to **PRG_TYPE**.
    :type prg_type: str, optional
    :return: The result of the prefix parity query as a RingTensor.
    :rtype: RingTensor

    .. important::
        The results of this method are different from the usual ones where 1 is obtained when the value is less than *alpha*,
        instead, 1 is obtained only when the value is equal to or greater than *alpha*.

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
