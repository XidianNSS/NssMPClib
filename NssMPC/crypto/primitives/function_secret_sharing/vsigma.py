"""
This document defines verifiable Sigma class based on the method used for DCF efficiently,
meaning that the correctness and security of the key generation stage (gen) and the evaluation stage (eval) are guaranteed.
The implementation is based on the work of Leo de Castro, Antigoni Polychroniadou. Lightweight, Maliciously Secure Verifiable Function Secret Sharing. EUROCRYPT 2022, LNCS 13275, pp. 150–179, 2022.
and SIGMA: Secure GPT Inference with Function Secret Sharing
For reference, see the `VFSS paper <https://doi.org/10.1007/978-3-031-06944-4_6>`_ and `Sigma paper <https://eprint.iacr.org/2023/1269>`_.
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.config import DEBUG_LEVEL, HALF_RING
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.vsigma_key import VSigmaKey
from NssMPC.crypto.primitives.function_secret_sharing.vdpf import VDPF


class VSigma(object):
    """
    FSS for verifiable sigma (VSigma).

    This class implements the verifiable sigma (VSigma), including the generation of keys and evaluation
    of the results. The emphasis is in ensuring the correctness of the key generation stage (gen) and the evaluation stage (eval).
    """

    @staticmethod
    def gen(num_of_keys):
        """
        Generate keys for VSigma.

        This method can generate multiple keys for VSigma, which can be used for the following evaluation.

        :param num_of_keys: Number of keys to generate.
        :type num_of_keys: int
        :return: The keys of all the involved parties (two parties).
        :rtype: tuple
        """
        return VSigmaKey.gen(num_of_keys)

    @staticmethod
    def eval(x_shift, keys, party_id):
        """
        Evaluate the output share for one party.

        According to the input x_shift, the party calculates the function value locally,
        which is the shared value of the original VSigma.
        And return the verification mark of the gen and eval process, thus ensuring the security.

        :param x_shift: The input to be evaluated.
        :type x_shift: RingTensor
        :param keys: The FSS key used for evaluation.
        :type keys: VSigmaKey or tuple
        :param party_id: The party ID (0 or 1), identifying which party is performing the evaluation.
        :type party_id: int
        :return: The tuple including the shared result of VSigma and the verification mark.
        :rtype: Tuple
        """
        return verifiable_sigma_eval(party_id, keys, x_shift)

    @staticmethod
    def cmp_eval(x, keys, party_id):
        from NssMPC import ArithmeticSecretSharing
        if DEBUG_LEVEL == 2:
            x_shift = ArithmeticSecretSharing(keys.r_in) + x
        else:
            x_shift = ArithmeticSecretSharing(keys.r_in.reshape(x.shape)) + x
        x_shift = x_shift.restore()
        return verifiable_sigma_eval(party_id, keys, x_shift)


def verifiable_sigma_eval(party_id, key, x_shift):
    """
    Evaluate the output share for one party.

    According to the input x_shift, the party calculates the function value locally,
    which is the shared value of the original VSigma.
    And return the verification mark of the gen and eval process, thus ensuring the security.

    :param x_shift: The input to be evaluated.
    :type x_shift: RingTensor
    :param key: The FSS key used for evaluation.
    :type key: VSigmaKey or tuple
    :param party_id: The party ID (0 or 1), identifying which party is performing the evaluation.
    :type party_id: int
    :return: The tuple including the shared result of VSigma and the verification mark.
    :rtype: Tuple
    """
    shape = x_shift.shape
    x_shift = x_shift.reshape(-1, 1)
    K, c = key
    y = x_shift % (HALF_RING - 1)
    y = y + 1
    out, pi = VDPF.ppq(y, K, party_id)
    out = x_shift.signbit() * party_id ^ c.reshape(-1, 1) ^ out
    return out.reshape(shape), pi
