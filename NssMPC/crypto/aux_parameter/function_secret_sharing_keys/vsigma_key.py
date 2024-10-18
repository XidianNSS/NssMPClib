"""
This document defines verifiable Sigma key class used for VSigma to guarantee the security for the gen and eval process.
The implementation is based on the work of Leo de Castro, Antigoni Polychroniadou. Lightweight, Maliciously Secure Verifiable Function Secret Sharing. EUROCRYPT 2022, LNCS 13275, pp. 150–179, 2022.
and SIGMA: Secure GPT Inference with Function Secret Sharing
For reference, see the `VFSS paper <https://doi.org/10.1007/978-3-031-06944-4_6>`_ and `Sigma paper <https://eprint.iacr.org/2023/1269>`_.
"""

#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch

from NssMPC import RingTensor
from NssMPC.config import HALF_RING, DEVICE
from NssMPC.crypto.aux_parameter import Parameter
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.vdpf_key import VDPFKey
from NssMPC.crypto.primitives.function_secret_sharing.vdpf import VDPF


class VSigmaKey(Parameter):
    """
    The FSS key class for verifiable sigma (VSigma).

    This class implements the generation method for the key used for VSigma evaluation.
    It includes methods for generating and managing the parameters used in VSigma protocol.

    ATTRIBUTES:
        * **dpf_key** (*VDPFKey*): The key of VSigma.
        * **c** (*RingTensor*): The parameter for offline support.
        * **r_in** (*RingTensor*): The offset of the function, for the purpose of blind the input.
    """

    def __init__(self, ver_dpf_key=VDPFKey()):
        """
        Initialize the VSigmaKey object.

        This method initializes the key `dpf_key` to *DPFKey* object, the MSB of offset `c` to *None*,
        and the offset `r_in` to *None*.
        """
        self.ver_dpf_key = ver_dpf_key
        self.c = None
        self.r_in = None

    def __iter__(self):
        """
        Return an iterator that contains the three attributes of the class.

        This method allows you to access the three attributes of a class instance in sequence as it is iterated over.
        :return: An iterator that contains the three attributes of the class.
        :rtype: iterator
        """
        return iter([self.r_in, self.ver_dpf_key, self.c])

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
        return verifiable_sigma_gen(num_of_keys)


# verifiable MSB protocol from sigma protocol without r_out
def verifiable_sigma_gen(num_of_keys):
    """
    Generate keys for VSigma.

    This method can generate multiple keys for VSigma, which can be used for the following evaluation.

    :param num_of_keys: Number of keys to generate.
    :type num_of_keys: int
    :return: The keys of all the involved parties (two parties).
    :rtype: tuple
    """
    r_in = RingTensor.random([num_of_keys])
    x1 = r_in
    y1 = r_in % (HALF_RING - 1)
    k0, k1 = VDPF.gen(num_of_keys, y1, RingTensor.convert_to_ring(1))
    c = x1.signbit() ^ 1
    c0 = torch.randint(0, 1, [num_of_keys], device=DEVICE)
    c0 = RingTensor.convert_to_ring(c0)
    c1 = c ^ c0

    k0 = VSigmaKey(k0)
    k1 = VSigmaKey(k1)

    k0.c = c0
    k1.c = c1

    from NssMPC import ArithmeticSecretSharing
    k0.r_in, k1.r_in = ArithmeticSecretSharing.share(r_in, 2)

    return k0, k1
