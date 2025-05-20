"""
This document defines function secret sharing for distributed interval comparison functions(DICF)
in secure two-party computing. We implement three different methods to compute the DICF under FSS scheme
, which respectively correspond to class DICF, GrottoDICF and SigmaDICF.

    .. note::
        Distributed point function:
            f(x)=1, if  down_bound <= x <= upper_bound; f(x)=0, else

"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.common.ring import RingTensor
from NssMPC.config.configs import PRG_TYPE, HALF_RING, data_type
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dicf_key import DICFKey, GrottoDICFKey, SigmaDICFKey
from NssMPC.crypto.primitives.function_secret_sharing.dcf import DCF
from NssMPC.crypto.primitives.function_secret_sharing.dpf import prefix_parity_query


class DICF:
    """
    **FSS for distributed interval comparison function (DICF)**

    This class implements the generation of keys and evaluation for computing DICF securely using function secret sharing.

    The implementation of class **DICF** is based on the work of
    *E. Boyle e.t.c. Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation.2021*.
    For reference, see the `paper <https://link.springer.com/chapter/10.1007/978-3-030-77886-6_30>`_.
    """

    @staticmethod
    def gen(num_of_keys, down_bound=RingTensor(0), upper_bound=RingTensor(HALF_RING - 1)):
        """
        Generate DICF keys.

        This method generates the DICF keys required for secure comparison.

        :param num_of_keys: The number of keys to generate.
        :type num_of_keys: int
        :param down_bound: The down bound of the interval for DICF, default is 0.
        :type down_bound: RingTensor
        :param upper_bound: The upper bound of the interval for DICF, default is *HALF_RING-1*.
        :type upper_bound: RingTensor
        :return: A tuple containing two DICFKey objects for the two parties.
        :rtype: Tuple[DICFKey, DICFKey]
        """
        return DICFKey.gen(num_of_keys, down_bound, upper_bound)

    @staticmethod
    def eval(x_shift, keys, party_id, down_bound=RingTensor(0), upper_bound=RingTensor(HALF_RING - 1)):
        """
        Evaluate the DICF on input *x_shift*.

        With this method, the party evaluates the function value locally using the input *x_shift*.
        The main principle of this method is to implement DICF with DCF twice.

        :param x_shift: The input RingTensor on which the DICF is evaluated.
        :type x_shift: RingTensor
        :param keys: The secret sharing keys required for evaluation.
        :type keys: DICFKey
        :param party_id: The party ID (0 or 1), identifying which party is performing the evaluation.
        :type party_id: int
        :param down_bound: The down bound of the interval for DICF, default is 0.
        :type down_bound: RingTensor
        :param upper_bound: The upper bound of the interval for DICF, default is *HALF_RING-1*.
        :type upper_bound: RingTensor
        :return: The sharing value of the comparison result for each party as a RingTensor.
        :rtype: RingTensor
        """
        p = down_bound
        q = upper_bound

        q1 = q + 1

        xp = (x_shift + (-1 - p))
        xq1 = (x_shift + (-1 - q1))

        s_p = DCF.eval(xp, keys.dcf_key, party_id, prg_type=PRG_TYPE)
        s_q = DCF.eval(xq1, keys.dcf_key, party_id, prg_type=PRG_TYPE)

        res = party_id * (((x_shift > p) + 0) - ((x_shift > q1) + 0)) - s_p + s_q + keys.z
        return res.to(data_type)


class GrottoDICF:
    """
    **FSS for distributed interval comparison function (DICF)** adapted from `Grotto <https://eprint.iacr.org/2023/108>`_.

    This class implements the generation of keys and evaluation for computing DICF securely using method adapted from Grotto.
    The main feature of this method is the use of a Parity-segment tree to improve computational efficiency.

    The implementation of class **GrottoDICF** is mainly based on the work of
    *Storrier K, Vadapalli A, Lyons A, et al. Grotto: Screaming fast (2+ 1)-PC for ℤ2n via (2, 2)-DPFs[J].IACR Cryptol. ePrint Arch., 2023, 2023: 108*.
    For reference, see the `paper <https://eprint.iacr.org/2023/108>`_.
    """

    @staticmethod
    def gen(num_of_keys: int, beta=RingTensor(1)):
        """
        Generate GrottoDICF keys.

        This method generates the DICF keys required for secure comparison with method adapted from Grotto.

        :param num_of_keys: The number of keys to generate.
        :type num_of_keys: int
        :param beta: The output value if the comparison is true.
        :type beta: RingTensor
        :return: A tuple containing two GrottoDICFKey objects for the two parties.
        :rtype: Tuple[GrottoDICFKey, GrottoDICFKey]
        """
        return GrottoDICFKey.gen(num_of_keys, beta)

    @staticmethod
    def eval(x_shift: RingTensor, key, party_id, prg_type=PRG_TYPE, down_bound=RingTensor(0),
             upper_bound=RingTensor(HALF_RING - 1)):
        """
        Evaluate the DICF on input *x_shift* with the method adapted from Grotto.

        With this method, the party evaluates the function value locally using the input *x_shift*.

        .. note::
            这里强调对超环情况的解决。ans和 tau处


        :param x_shift: The input RingTensor on which the GrottoDICF is evaluated. x_shift的解释是否正确
        :type x_shift: RingTensor
        :param key: The secret sharing keys required for evaluation.
        :type key: GrottoDICFKey
        :param party_id: The party ID (0 or 1), identifying which party is performing the evaluation.
        :type party_id: int
        :param prg_type: The type of pseudorandom generator (PRG) used during evaluation, defaults to PRG_TYPE.
        :type prg_type: str, optional
        :param down_bound: The down bound of the interval for GrottoDICF, default is 0.
        :type down_bound: RingTensor
        :param upper_bound: The upper bound of the interval for GrottoDICF, default is *HALF_RING-1*.
        :type upper_bound: RingTensor
        :return: The sharing value of the comparison result for each party as a RingTensor.
        :rtype: RingTensor
        """
        p = down_bound + x_shift
        q = upper_bound + x_shift
        cond = (p ^ q) < 0
        tau = ((p > q) ^ cond) * party_id
        x = RingTensor.stack([p, q]).view(2, -1, 1)

        parity_x = prefix_parity_query(x, key.dpf_key, party_id, prg_type)

        ans = (parity_x[0] ^ parity_x[1]).view(x_shift.shape) ^ tau

        return ans


class SigmaDICF:
    """
    **FSS for distributed interval comparison function (DICF)** adapted from `Sigma <https://eprint.iacr.org/2023/1269>`_.

    This class implements the generation of keys and evaluation for computing DICF securely using method adapted from Sigma.
    The main feature of this method is the use of a single evaluation of DPF-based comparison to improve computational efficiency.

    The implementation of class **SigmaDICF** is mainly based on the work of
    *SIGMA: Secure GPT Inference with Function Secret Sharing*.
    For reference, see the `paper <https://eprint.iacr.org/2023/1269>`_.
    """

    @staticmethod
    def gen(num_of_keys):
        """
        Generate SigmaDICF keys.

        This method generates the DICF keys required for secure comparison with method adapted from Sigma.

        :param num_of_keys: The number of keys to generate.
        :type num_of_keys: int
        :return: A tuple containing two SigmaDICFKey objects for the two parties.
        :rtype: Tuple[SigmaDICFKey, SigmaDICFKey]
        """
        return SigmaDICFKey.gen(num_of_keys)

    @staticmethod
    def eval(x_shift: RingTensor, key, party_id):
        """
        Evaluate the DICF on input *x_shift* with the method adapted from Sigma.

        With this method, the party evaluates the function value locally using the input *x_shift*.

        :param x_shift: The input RingTensor on which the SigmaDICF is evaluated.
        :type x_shift: RingTensor
        :param key: The secret sharing keys required for evaluation.
        :type key: SigmaDICFKey
        :param party_id: The party ID (0 or 1), identifying which party is performing the evaluation.
        :type party_id: int
        :return: The sharing value of the comparison result for each party as a RingTensor.
        :rtype: RingTensor
        """
        shape = x_shift.shape
        x_shift = x_shift.view(-1, 1)
        y = x_shift % (2 ** (x_shift.bit_len - 1) - 1)
        y = y + 1
        out = prefix_parity_query(y, key.dpf_key, party_id)
        out = x_shift.signbit() * party_id ^ key.c.view(-1, 1) ^ out
        return out.view(shape)

    @staticmethod
    def one_key_eval(input_list, key, party_id):
        """
        Evaluate multiple inputs with one key, can be used only when the input data is the offset of the same number.

        The difference between this method and :meth:`~NssMPC.crypto.primitives.function_secret_sharing.dicf.SigmaDICF.eval`
        is that this method can evaluate **multiple inputs** with only one key.

        :param input_list: The input RingTensor on which the SigmaDICF is evaluated.
        :type input_list: list[RingTensor]
        :param key: The secret sharing keys required for evaluation.
        :type key: SigmaDICFKey
        :param party_id: The party ID (0 or 1), identifying which party is performing the evaluation.
        :type party_id: int
        :return: The sharing value of the comparison result for each party as a RingTensor.
        :rtype: RingTensor
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
