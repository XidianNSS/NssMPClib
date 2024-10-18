"""
This document defines verifiable distributed point function (VDPF),
meaning that the correctness and security of the key generation stage (gen) and the evaluation stage (eval) are guaranteed.
The implementation is based on the work of Leo de Castro, Antigoni Polychroniadou. Lightweight, Maliciously Secure Verifiable Function Secret Sharing. EUROCRYPT 2022, LNCS 13275, pp. 150–179, 2022.
For reference, see the `paper <https://doi.org/10.1007/978-3-031-06944-4_6>`_.

.. tip::
    distributed point function (DPF)
    f(x)=b, if x = α; f(x)=0, else

"""

#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC import RingTensor
from NssMPC.common.random import PRG
from NssMPC.common.utils import convert_tensor
from NssMPC.config import DEVICE, LAMBDA, PRG_TYPE
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys import CW
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.vdpf_key import VDPFKey


class VDPF(object):
    """
    FSS for verifiable distributed point function (VDPF).

    This class implements the verifiable distributed point function (VDPF), including the generation of keys and evaluation
    of the results. The emphasis is in ensuring the correctness of the key generation stage (gen) and the evaluation stage (eval).
    """

    @staticmethod
    def gen(num_of_keys, alpha, beta):
        """
        Generate keys for VDPF.

        This method can generate multiple keys for VDPF, which can be used for the following evaluation.

        :param num_of_keys: Number of keys to generate.
        :type num_of_keys: int
        :param alpha: The comparison point of VDPF.
        :type alpha: RingTensor
        :param beta: The output value if the comparison is true.
        :type beta: RingTensor
        :return: The keys of all the involved parties (two parties).
        :rtype: tuple
        """
        return VDPFKey.gen(num_of_keys, alpha, beta)

    @staticmethod
    def eval(x, keys, party_id):
        """
        Evaluate the output share for one party.

        According to the input x, the party calculates the function value locally,
        which is the shared value of the original VDPF.
        And return the verification mark of the gen and eval process, thus ensuring the security.

        :param x: The input to be evaluated.
        :type x: RingTensor
        :param keys: The FSS key used for evaluation.
        :type keys: VDPFKey
        :param party_id: The party ID (0 or 1), identifying which party is performing the evaluation.
        :type party_id: int
        :return: The tuple including the shared result of VDPF and the verification mark.
        :rtype: Tuple
        """
        shape = x.shape
        x = x.view(-1, 1)
        res, h = vdpf_eval(x, keys, party_id)
        return res.reshape(shape), h

    @staticmethod
    def one_key_eval(x, keys, num, party_id):
        """
        Evaluate multiple inputs use only one VDPF key.

        According to the input x, the party calculates the function value locally,
        which is the shared value of the original VDPF.
        The emphasis is in evaluating multiple inputs use only one key.

        :param x: The input to be evaluated.
        :type x: RingTensor
        :param keys: The FSS key used for evaluation.
        :type keys: VDPFKey
        :param num: The number of input to be evaluated.
        :type num: int
        :param party_id: The party ID (0 or 1), identifying which party is performing the evaluation.
        :type party_id: int
        :return: The tuple including the shared result of VDPF and the verification mark.
        :rtype: Tuple
        """
        shape = x.shape
        x = x.view(num, -1, 1)
        res, h = vdpf_eval(x, keys, party_id)
        return res.reshape(shape), h

    @staticmethod
    def ppq(x, keys, party_id):
        """
        Implement the verifiable prefix parity query.

        This method implements a verifiable function on top of the original ppq method,
        ensuring the correctness of evaluation stage.
        For more information on ppq, please click `here <NssMPC.crypt.primitives.function_secret_sharing.dpf.prefix_parity_query>`.

        :param x: The input to be evaluated.
        :type x: RingTensor
        :param keys: The FSS key used for evaluation.
        :type keys: VDPFKey
        :param party_id: The party ID (0 or 1), identifying which party is performing the evaluation.
        :type party_id: int
        :return: The tuple including the shared result of VDPF and the verification mark.
        :rtype: Tuple
        """
        return ver_ppq_dpf(x, keys, party_id)


def vdpf_eval(x: RingTensor, keys: VDPFKey, party_id):
    """
    Evaluate the output share for one party.

    According to the input x, the party calculates the function value locally,
    which is the shared value of the original VDPF.

    :param x: The input to be evaluated.
    :type x: RingTensor
    :param keys: The FSS key used for evaluation.
    :type keys: VDPFKey
    :param party_id: The party ID (0 or 1), identifying which party is performing the evaluation.
    :type party_id: int
    :return: The tuple including the shared result of VDPF and the verification mark.
    :rtype: Tuple
    """
    prg = PRG(PRG_TYPE, DEVICE)

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

    seed = s_last + x.tensor
    # seed = torch.cat((s_last, x.tensor), dim=1)

    # TODO: Hash function
    prg.set_seeds(seed.transpose(1, 2))
    pi_ = prg.bit_random_tensor(4 * LAMBDA)
    # t_last = s_last & 1

    dpf_result = pow(-1, party_id) * (convert_tensor(s_last) + t_last * keys.ocw)

    seed = keys.cs ^ (pi_ ^ (keys.cs * t_last))
    # prg.set_seeds(seed[:, 0:2])
    prg.set_seeds(seed.transpose(1, 2))
    h_ = prg.bit_random_tensor(2 * LAMBDA)
    # pi = keys.cs ^ h_

    return dpf_result, h_.sum(dim=1)


def ver_ppq_dpf(x, keys, party_id):
    """
    Implement the verifiable prefix parity query.

    This method implements a verifiable function on top of the original ppq method,
    ensuring the correctness of evaluation stage.
    For more information on ppq, please click :meth:`~NssMPC.crypt.primitives.function_secret_sharing.dpf.prefix_parity_query`.

    :param x: The input to be evaluated.
    :type x: RingTensor
    :param keys: The FSS key used for evaluation.
    :type keys: VDPFKey
    :param party_id: The party ID (0 or 1), identifying which party is performing the evaluation.
    :type party_id: int
    :return: The tuple including the shared result of verifiable ppq and the verification mark.
    :rtype: Tuple
    """
    # 将输入展平
    shape = x.tensor.shape
    x = x.clone()
    x.tensor = x.tensor.view(-1, 1)

    d = 0
    psg_b = 0
    t_last = party_id
    s_last = keys.s
    prg = PRG(PRG_TYPE, DEVICE)

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

    prg = PRG(PRG_TYPE, DEVICE)

    seed = s_last + x.tensor
    # seed = torch.cat((s_last, x.tensor), dim=1)
    prg.set_seeds(seed)
    pi_ = prg.bit_random_tensor(4 * LAMBDA)
    seed = keys.cs ^ (pi_ ^ (keys.cs * t_last))  # TODO: HASH
    prg.set_seeds(seed.transpose(-2, -1))
    h_ = prg.bit_random_tensor(2 * LAMBDA)
    h_ = pi_[..., :2 * LAMBDA]
    pi = RingTensor.convert_to_ring(h_.sum(dim=1))
    # pi = RingTensor.convert_to_ring(pi_.sum(dim=1))
    return RingTensor(psg_b.view(shape)), pi
