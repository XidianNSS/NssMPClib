from typing import List

import torch

from NssMPC.config import *
from NssMPC.config import PRG_TYPE, LAMBDA, DEVICE, HALF_RING, data_type, DEBUG_LEVEL
from NssMPC.infra.mpc.aux_parameter import ParamProvider
from NssMPC.infra.mpc.aux_parameter.parameter import Parameter
from NssMPC.infra.mpc.party import Party3PC
from NssMPC.infra.prg import PRG
from NssMPC.infra.tensor import RingTensor
from NssMPC.infra.utils import convert_tensor
from NssMPC.primitives.secret_sharing import AdditiveSecretSharing


class CW(object):
    """
    A class for correction words (cw), which is used for FSS.

    This class is used for the calculation of DPF, DCF, etc. And is part of the `keys` required for these calculations.
    And it defines the operations related to the CW class properties as well as the generation of cw.
    """

    def __init__(self, **kwargs):
        """
        Initialize an instance of the CW class and directly update all passed keyword arguments to instance properties.

        This method allows for flexible setting of any properties at the time of creating a class instance,
        without the need to predefine them in the class.

        Args:
            **kwargs: The attributes that need to be set.

        Examples:
            >>> cw = CW(attr=[1, 2, 3])
        """
        self.__dict__.update(kwargs)

    def __setitem__(self, key, value):
        """
        Assign values to attributes of CW instances by index or key.

        Iterate over all the properties of the object, and if they support an index or key assignment,
        they are dynamically updated based on the key and value (the CW instance that contains the value to assign to) passed in.

        Args:
            key: The key or index of each property to assign to.
            value (CW): The CW instance containing the value of the property to be assigned.

        Returns:
            CW: The CW instance after updating the property value.

        Examples:
            >>> cw[0] = other_cw
        """
        for k, v in self.__dict__.items():
            if hasattr(v, '__setitem__'):
                v[key] = getattr(value, k)

    def __getitem__(self, item):
        """
        Get the value in the CW object by index or key.

        Iterate over all the properties of the object, try to extract values from those that support index or key assignment,
        and store them in a new CW object and return it.

        Args:
            item: The key or index you want to use to retrieve the value.

        Returns:
            CW: The CW instance after obtaining the corresponding property value.

        Examples:
            >>> val = cw[0]
        """
        ret = CW()
        for k, v in self.__dict__.items():
            if hasattr(v, '__getitem__'):
                setattr(ret, k, v[item])
        return ret

    def to(self, device):
        """
        Transfers the properties inside the CW object that support the `to` operation to the specified device(`CPU` or `GPU`).

        This method iterates over all properties of the object, and if a property supports the `to` method,
        it is transferred to the target device.

        Args:
            device (str): The target device to transfer to.

        Returns:
            CW: The CW object after transferring the device.

        Examples:
            >>> cw.to('cuda')
        """
        for k, v in self.__dict__.items():
            if isinstance(v, (RingTensor, torch.Tensor)) or hasattr(v, 'to'):
                setattr(self, k, v.to(device))
        return self

    @staticmethod
    def gen_dcf_cw(prg: torch.classes.csprng_aes.AES_PRG, new_seeds, lmd: int, bit_len: int = BIT_LEN):
        """
        Generate the Correction Words (CW) for the Distributed Comparison Function (DCF).

        This method computes the correction words required for the GGM tree expansion in the DCF protocol.
        It expands the current seeds using the AES-based pseudo-random generator to produce the next layer's seeds and control bits.

        Note:
            This implementation specifically requires a PRG with an **AES kernel** for security and performance alignment.

        Args:
            prg: The AES-based pseudorandom generator instance.
            new_seeds (RingTensor or torch.Tensor): The input seeds for the current layer of the GGM tree.
            lmd: The security parameter lambda, representing the bit length of the seeds (e.g., 128).
            bit_len: The domain size (bit length) of the input to be compared. Defaults to global BIT_LEN.

        Returns:
            tuple: A tuple containing the components of the Correction Words (e.g., seeds correction, payload correction).

        Raises:
            ValueError: If the PRG kernel is not 'AES'.

        Examples:
            >>> s_l, v_l, t_l, s_r, v_r, t_r = CW.gen_dcf_cw(prg, seeds, 128)
        """
        prg.set_seeds(new_seeds)
        random_bits = prg.bit_random_tensor(4 * lmd + 2)
        s_num = 128 // bit_len
        s_l_res = random_bits[..., 0:s_num]
        v_l_res = random_bits[..., s_num: s_num + s_num]

        s_r_res = random_bits[..., 2 * s_num: 2 * s_num + s_num]

        v_r_res = random_bits[..., 3 * s_num: 3 * s_num + s_num]

        t_l_res = random_bits[..., 4 * s_num] & 1
        t_l_res = t_l_res.unsqueeze(-1)
        t_r_res = random_bits[..., 4 * s_num] >> 1 & 1
        t_r_res = t_r_res.unsqueeze(-1)
        return s_l_res, v_l_res, t_l_res, s_r_res, v_r_res, t_r_res
        # else:
        #     raise ValueError("kernel is not supported!")


class CWList(list):
    """
    A list-like collection for CW instances used in FSS (Function Secret Sharing) computations.

    This class is used to aggregate the CW instances involved in FSS computations and defines
    corresponding operations on each CW in the list to reduce the complexity of the code during usage.
    """

    def __init__(self, *args):
        """
        Initialize a CWList instance.

        Args:
            *args: Variable length argument list.

        Examples:
            >>> cw_list = CWList(cw1, cw2)
        """
        super().__init__(args)

    def getitem(self, item):
        """
        Retrieve the value corresponding to the item from each element in the list.

        This method iterates over each element (CW instance) in the CWList,
        calls the indexing operation on these elements, and stores the result in a new CWList instance to return.
        It is equivalent to calling the __getitem__ method on each CW instance.

        Args:
            item: The key or index you want to use to retrieve the value.

        Returns:
            CWList: A new CWList instance containing the corresponding values from each CW instance.

        Examples:
            >>> sub_list = cw_list.getitem(0)
        """
        ret = CWList()
        for element in self:
            ret.append(element[item])
        return ret

    def setitem(self, item, value):
        """
        Set the value corresponding to the item in each CW instance with the corresponding values in the value parameter.

        This method iterates over each element (CW instance) in the CWList,
        calls the assignment operation with the given item on these elements, and stores the result.
        It is equivalent to calling the __setitem__ method on each CW instance.

        Args:
            item: The key or index you want to use to assign the value.
            value (CWList): The CWList instance containing the value of the property to be assigned.

        Returns:
            CWList: The CWList object after transferring the instances to the specified device.

        Examples:
            >>> cw_list.setitem(0, other_cw_list)
        """
        for i in range(len(self)):
            self[i][item] = value[i]

    def to(self, device):
        """
        Transfers the CW instance inside the CWList object that support the `to` operation to the specified device(`CPU` or `GPU`).

        This method iterates over all items (suppose to be CW instances) of the object, and if a item supports the `to` method,
        it is transferred to the target device.

        Args:
            device (str): The target device to transfer to.

        Returns:
            CWList: The CWList object after transferring the device.

        Examples:
            >>> cw_list.to('cuda')
        """
        temp = CWList()
        for v in self:
            if isinstance(v, CW) or isinstance(v, (RingTensor, torch.Tensor)) or hasattr(v, 'to'):
                temp.append(v.to(device))
        return temp

    def expand_as(self, input):
        """
        Expands each element in CWList to be the same shape as the corresponding element in another input sequence.

        Args:
            input (list or torch.Tensor): input sequence

        Returns:
            CWList: A new CWList instance containing the extended element

        Examples:
            >>> expanded_list = cw_list.expand_as(input_tensor)
        """
        ret = CWList()
        for i, value in enumerate(self):
            if hasattr(value, 'expand_as'):
                ret.append(value.expand_as(input[i]))
        return ret


class DPFKey(Parameter):
    """
    The function secret sharing key for distributed point function(DPF).

    This class implements the secret sharing keys required for function secret sharing(FSS) in
    distributed point functions (DPF). It includes methods for generating
    and managing the correction words and other key parameters used in the DPF protocol.

    Attributes:
        s (torch.Tensor): A binary string (lambda bits) generated by the PRG for the root node.
        cw_list (CWList): The list of correction words used in the comparison function.
        ex_cw_DPF (torch.Tensor): The extra check word used for DPF calculation.
        size (int): The size of the key (default is 0).
    """

    def __init__(self):
        """
        Initialize the DPFKey object.

        This method initializes the seed `s` to **None**, the list of correction words `cw_list` to a **CWList** object,
        the extra correction word `ex_cw_DPF` to **None**, and sets the size of the key to 0.

        Examples:
            >>> key = DPFKey()
        """
        self.s = None
        self.cw_list = CWList()
        self.ex_cw_dpf = None
        self.size = 0

    @staticmethod
    def gen(num_of_keys, alpha, beta):
        """
        Generate DPF keys.

        This method generates multiple DPF keys required for secure comparison.

        Note:
            Distributed point function:
                f(x)=beta, if x < alpha; f(x)=0, else

        Args:
            num_of_keys (int): The number of keys to generate.
            alpha (RingTensor): The comparison point (private value for comparison).
            beta (RingTensor): The output value if the comparison is true.

        Returns:
            Tuple[DPFKey, DPFKey]: A tuple containing two DPFKey objects for the two parties.

        Examples:
            >>> k0, k1 = DPFKey.gen(10, alpha, beta)
        """
        seed_0 = torch.randint(-HALF_RING, HALF_RING - 1, [num_of_keys, LAMBDA // BIT_LEN], dtype=data_type,
                               device=DEVICE)
        seed_1 = torch.randint(-HALF_RING, HALF_RING - 1, [num_of_keys, LAMBDA // BIT_LEN], dtype=data_type,
                               device=DEVICE)

        prg = torch.classes.csprng_aes.AES_PRG()
        prg.set_seeds(seed_0)
        s_0_0 = prg.bit_random(LAMBDA)
        prg.set_seeds(seed_1)
        s_0_1 = prg.bit_random(LAMBDA)

        k0 = DPFKey()
        k1 = DPFKey()

        k0.s = s_0_0
        k1.s = s_0_1

        s_last_0 = s_0_0
        s_last_1 = s_0_1

        t_last_0 = 0
        t_last_1 = 1

        for i in range(alpha.bit_len):
            s_l_0, t_l_0, s_r_0, t_r_0 = gen_dpf_cw(prg, s_last_0, LAMBDA)
            s_l_1, t_l_1, s_r_1, t_r_1 = gen_dpf_cw(prg, s_last_1, LAMBDA)

            cond = (alpha.get_tensor_bit(alpha.bit_len - 1 - i) == 0).view(-1, 1)

            l_tensors = [s_l_0, s_l_1, t_l_0, t_l_1]
            r_tensors = [s_r_0, s_r_1, t_r_0, t_r_1]

            keep_tensors = [torch.where(cond, l, r) for l, r in zip(l_tensors, r_tensors)]
            lose_tensors = [torch.where(cond, r, l) for l, r in zip(l_tensors, r_tensors)]

            s_keep_0, s_keep_1, t_keep_0, t_keep_1 = keep_tensors
            s_lose_0, s_lose_1, t_lose_0, t_lose_1 = lose_tensors

            s_cw = s_lose_0 ^ s_lose_1

            t_l_cw = t_l_0 ^ t_l_1 ^ ~cond ^ 1
            t_r_cw = t_r_0 ^ t_r_1 ^ ~cond

            cw = CW(s_cw=s_cw, t_cw_l=t_l_cw, t_cw_r=t_r_cw, lmd=LAMBDA)

            k0.cw_list.append(cw)
            k1.cw_list.append(cw)

            t_keep_cw = torch.where(cond, t_l_cw, t_r_cw)

            s_last_0 = s_keep_0 ^ (t_last_0 * s_cw)
            s_last_1 = s_keep_1 ^ (t_last_1 * s_cw)

            t_last_0 = t_keep_0 ^ (t_last_0 * t_keep_cw)
            t_last_1 = t_keep_1 ^ (t_last_1 * t_keep_cw)
        k0.ex_cw_dpf = k1.ex_cw_dpf = pow(-1, t_last_1) * (
                beta.tensor
                - convert_tensor(s_last_0)
                + convert_tensor(s_last_1))

        return k0, k1


class DPF:
    """
    **FSS for Distributed Point Function (DPF)**.

    This class implements the generation and evaluation of keys for computing DPF securely using function secret sharing.
    """

    @staticmethod
    def gen(num_of_keys: int, alpha: RingTensor, beta: RingTensor):
        """
        Generate DPF keys.

        This method generates the DPF keys required for secure comparison.

        Note:
            Distributed point function:
                f(x)=beta, if x = alpha; f(x)=0, else

        Args:
            num_of_keys: The number of keys to generate.
            alpha: The comparison point (private value for comparison).
            beta: The output value if the comparison is true.

        Returns:
            Tuple[DPFKey, DPFKey]: A tuple containing two DPFKey objects for the two parties.

        Examples:
            >>> k0, k1 = DPF.gen(10, alpha, beta)
        """
        return DPFKey.gen(num_of_keys, alpha, beta)

    @staticmethod
    def eval(x: RingTensor, keys: DPFKey, party_id: int, prg_type=PRG_TYPE):
        """
        Evaluate the DPF on input `x`.

        With this method, the party evaluates the function value locally using the input *x*,
        provided keys and party ID. It performs bitwise operations to securely
        compute the comparison result, which is the **shared value** of the original result.

        Args:
            x (RingTensor): The input RingTensor on which the DPF is evaluated.
            keys (DPFKey): The secret sharing keys required for evaluation.
            party_id (int): The party ID (0 or 1), identifying which party is performing the evaluation.
            prg_type (str, optional): The type of pseudorandom generator (PRG) used during evaluation, defaults to **PRG_TYPE**.

        Returns:
            RingTensor: The sharing value of the comparison result for each party as a RIngTensor.

        Examples:
            >>> res = DPF.eval(x, key, 0)
        """
        x = x.clone()

        prg = torch.classes.csprng_aes.AES_PRG()

        t_last = party_id

        s_last = keys.s

        for i in range(x.bit_len):
            cw = keys.cw_list[i]

            s_cw = cw.s_cw
            t_cw_l = cw.t_cw_l
            t_cw_r = cw.t_cw_r

            s_l, t_l, s_r, t_r = gen_dpf_cw(prg, s_last, LAMBDA)

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

    Important:
        We put this method in ``dpf`` document because the key used in this method is DPFKey, but actually it
        implements the functionality of DCF.

    Note:
        The implementation is based on the work of
        **Storrier, K., Vadapalli, A., Lyons, A., & Henry, R. (2023). Grotto: Screaming Fast (2 + 1)-PC for Z2ⁿ via (2, 2)-DPFs**.
        For reference, see the `paper <https://eprint.iacr.org/2023/108>`_.

    Args:
        x: The input RingTensor on which the prefix parity query is performed.
        keys (DPFKey): The secret sharing keys required for prefix parity query.
        party_id (int): The party ID (0 or 1), identifying which party is performing the evaluation.
        prg_type (str, optional): The type of pseudorandom generator (PRG) used during evaluation, defaults to **PRG_TYPE**.

    Returns:
        RingTensor: The result of the prefix parity query as a RingTensor.

    Important:
        The results of this method are different from the usual ones where 1 is obtained when the value is less than *alpha*,
        instead, 1 is obtained only when the value is equal to or greater than *alpha*.

    Examples:
        >>> res = prefix_parity_query(x, key, 0)
    """
    s = keys.s
    cw_list_s = [cw.s_cw for cw in keys.cw_list]
    cw_list_tl = [cw.t_cw_l for cw in keys.cw_list]
    cw_list_tr = [cw.t_cw_r for cw in keys.cw_list]
    return RingTensor(_prefix_parity_query(x.tensor, s, cw_list_s, cw_list_tl, cw_list_tr, party_id, x.bit_len, LAMBDA),
                      x.dtype, x.device)


# def prefix_parity_query(x: RingTensor, keys, party_id, prg_type=PRG_TYPE):
#     """
#     Return the prefix parity query of the input *x*, thus improve the efficiency of the evaluation process.
#
#     By transforming the distributed point function evaluation(EVAL) process
#     to computing the prefix parity sum of a section (Prefix Parity Sum), we can improve the computational efficiency.
#     So based on the input *x*, the participant locally computes the parity of the point in the Parity-segment tree,
#     and the result will be 0 if **x < alpha**, and 1 **otherwise**.
#
#     .. important::
#         We put this method in ``dpf`` document because the key used in this method is DPFKey, but actually it
#         implements the functionality of DCF.
#
#     .. note::
#         The implementation is based on the work of
#         **Storrier, K., Vadapalli, A., Lyons, A., & Henry, R. (2023). Grotto: Screaming Fast (2 + 1)-PC for Z2ⁿ via (2, 2)-DPFs**.
#         For reference, see the `paper <https://eprint.iacr.org/2023/108>`_.
#
#     :param x: The input RingTensor on which the prefix parity query is performed.
#     :type x: RingTensor
#     :param keys: The secret sharing keys required for prefix parity query.
#     :type keys: DPFKey
#     :param party_id: The party ID (0 or 1), identifying which party is performing the evaluation.
#     :type party_id: int
#     :param prg_type: The type of pseudorandom generator (PRG) used during evaluation, defaults to **PRG_TYPE**.
#     :type prg_type: str, optional
#     :return: The result of the prefix parity query as a RingTensor.
#     :rtype: RingTensor
#
#     .. important::
#         The results of this method are different from the usual ones where 1 is obtained when the value is less than *alpha*,
#         instead, 1 is obtained only when the value is equal to or greater than *alpha*.
#
#     """
#     prg = torch.classes.csprng_aes.AES_PRG()
#
#     d = 0
#     psg_b = 0
#     t_last = party_id
#
#     s_last = keys.s
#     for i in range(x.bit_len):
#         cw = keys.cw_list[i]
#
#         s_cw = cw.s_cw
#         t_cw_l = cw.t_cw_l
#         t_cw_r = cw.t_cw_r
#
#         s_l, t_l, s_r, t_r = gen_dpf_cw(prg, s_last, LAMBDA)
#
#         s1_l = s_l ^ (s_cw * t_last)
#         t1_l = t_l ^ (t_cw_l * t_last)
#         s1_r = s_r ^ (s_cw * t_last)
#         t1_r = t_r ^ (t_cw_r * t_last)
#
#         x_shift_bit = ((x.tensor >> (x.bit_len - 1 - i)) & 1)
#
#         cond = (d != x_shift_bit)
#
#         d = torch.where(cond, x_shift_bit, d)
#         psg_b = torch.where(cond, psg_b ^ t_last, psg_b)
#
#         s_last = torch.where(x_shift_bit.to(torch.bool), s1_r, s1_l)
#         t_last = torch.where(x_shift_bit.to(torch.bool), t1_r, t1_l)
#
#     psg_b = (psg_b ^ t_last) * d + psg_b * (1 - d)
#
#     return RingTensor(psg_b, x.dtype, x.device)

def _prefix_parity_query(x,
                         s,
                         cw_list_s: List[torch.Tensor],
                         cw_list_tl: List[torch.Tensor],
                         cw_list_tr: List[torch.Tensor],
                         party_id: int,
                         bit_len: int,
                         lmd: int):
    prg = torch.classes.csprng_aes.AES_PRG()

    d = torch.zeros_like(x)
    psg_b = torch.zeros_like(x)
    t_last = torch.full_like(x, party_id)

    s_last = s
    for i in range(bit_len):
        s_cw = cw_list_s[i]
        t_cw_l = cw_list_tl[i]
        t_cw_r = cw_list_tr[i]

        s_l, t_l, s_r, t_r = gen_dpf_cw(prg, s_last, lmd)

        s1_l = s_l ^ (s_cw * t_last)
        t1_l = t_l ^ (t_cw_l * t_last)
        s1_r = s_r ^ (s_cw * t_last)
        t1_r = t_r ^ (t_cw_r * t_last)

        x_shift_bit = ((x >> (bit_len - 1 - i)) & 1)

        cond = (d != x_shift_bit)

        d = torch.where(cond, x_shift_bit, d)
        psg_b = torch.where(cond, psg_b ^ t_last, psg_b)

        s_last = torch.where(x_shift_bit.to(torch.bool), s1_r, s1_l)
        t_last = torch.where(x_shift_bit.to(torch.bool), t1_r, t1_l)

    psg_b = (psg_b ^ t_last) * d + psg_b * (1 - d)

    return psg_b


class DCFKey(Parameter):
    """
    The function secret sharing key for distributed comparison function(DCF).

    This class implements the secret sharing keys required for function secret sharing(FSS) in
    distributed comparison functions (DCF). It includes methods for generating
    and managing the correction words and other key parameters used in the DCF protocol.

    Attributes:
        s (torch.Tensor): A binary string (lambda bits) generated by the PRG for the root node.
        cw_list (CWList): The list of correction words used in the comparison function.
        ex_cw_dcf (torch.Tensor): The extra check word used for DCF calculation.
        size (int): The size of the key (default is 0).
    """

    def __init__(self):
        """
        Initialize the DCFKey object.

        This method initializes the seed `s` to *None*, the list of correction words `cw_list` to a CWList object,
        the extra correction word `ex_cw_dcf` to *None*, and sets the size of the key to 0.

        Examples:
            >>> key = DCFKey()
        """
        self.s = None
        self.cw_list = CWList()
        self.ex_cw_dcf = None
        self.size = 0

    @staticmethod
    def gen(num_of_keys, alpha, beta):
        """
        Generate DCF keys.

        This method generates multiple DCF keys required for secure comparison.

        Note:
            Distributed point function:
                f(x)=beta, if x < alpha; f(x)=0, else

        Args:
            num_of_keys (int): The number of keys to generate.
            alpha (RingTensor): The comparison point (private value for comparison).
            beta (RingTensor): The output value if the comparison is true.

        Returns:
            Tuple[DCFKey, DCFKey]: A tuple containing two DCFKey objects for the two parties.

        Examples:
            >>> k0, k1 = DCFKey.gen(10, alpha, beta)
        """
        seed_0 = torch.randint(-HALF_RING, HALF_RING - 1, [num_of_keys, LAMBDA // BIT_LEN], dtype=data_type,
                               device=DEVICE)
        seed_1 = torch.randint(-HALF_RING, HALF_RING - 1, [num_of_keys, LAMBDA // BIT_LEN], dtype=data_type,
                               device=DEVICE)

        prg = PRG(PRG_TYPE, device=DEVICE)
        prg.set_seeds(seed_0)
        s_0_0 = prg.bit_random_tensor(LAMBDA)
        prg.set_seeds(seed_1)
        s_0_1 = prg.bit_random_tensor(LAMBDA)

        k0 = DCFKey()
        k1 = DCFKey()

        k0.s = s_0_0
        k1.s = s_0_1

        s_last_0 = s_0_0
        s_last_1 = s_0_1

        t_last_0 = torch.zeros(num_of_keys, 1, dtype=data_type, device=DEVICE)
        t_last_1 = torch.ones(num_of_keys, 1, dtype=data_type, device=DEVICE)

        v_a = torch.zeros((num_of_keys, 1), dtype=data_type, device=DEVICE)

        for i in range(alpha.bit_len):
            s_l_0, v_l_0, t_l_0, s_r_0, v_r_0, t_r_0 = CW.gen_dcf_cw(prg, s_last_0, LAMBDA)
            s_l_1, v_l_1, t_l_1, s_r_1, v_r_1, t_r_1 = CW.gen_dcf_cw(prg, s_last_1, LAMBDA)

            cond = (alpha.get_tensor_bit(alpha.bit_len - 1 - i) == 0).view(-1, 1)

            l_tensors = [s_l_0, s_l_1, v_l_0, v_l_1, t_l_0, t_l_1]
            r_tensors = [s_r_0, s_r_1, v_r_0, v_r_1, t_r_0, t_r_1]
            keep_tensors = [torch.where(cond, l, r) for l, r in zip(l_tensors, r_tensors)]
            lose_tensors = [torch.where(cond, r, l) for l, r in zip(l_tensors, r_tensors)]

            s_keep_0, s_keep_1, v_keep_0, v_keep_1, t_keep_0, t_keep_1 = keep_tensors
            s_lose_0, s_lose_1, v_lose_0, v_lose_1, t_lose_0, t_lose_1 = lose_tensors

            s_cw = s_lose_0 ^ s_lose_1

            v_cw = pow(-1, t_last_1) * (convert_tensor(v_lose_1) - convert_tensor(v_lose_0) - v_a)

            v_cw = torch.where(alpha.get_tensor_bit(alpha.bit_len - 1 - i) == 1, v_cw + pow(-1, t_last_1) * beta.tensor,
                               v_cw + torch.zeros_like(beta.tensor))

            v_a = (v_a
                   - convert_tensor(v_keep_1)
                   + convert_tensor(v_keep_0)
                   + pow(-1, t_last_1) * v_cw)

            t_l_cw = t_l_0 ^ t_l_1 ^ ~cond ^ 1
            t_r_cw = t_r_0 ^ t_r_1 ^ ~cond

            cw = CW(s_cw=s_cw, v_cw=v_cw, t_cw_l=t_l_cw, t_cw_r=t_r_cw, lmd=LAMBDA)

            k0.cw_list.append(cw)
            k1.cw_list.append(cw)

            t_keep_cw = torch.where(cond, t_l_cw, t_r_cw)

            s_last_0 = s_keep_0 ^ (t_last_0 * s_cw)
            s_last_1 = s_keep_1 ^ (t_last_1 * s_cw)

            t_last_0 = t_keep_0 ^ (t_last_0 * t_keep_cw)
            t_last_1 = t_keep_1 ^ (t_last_1 * t_keep_cw)

        k0.ex_cw_dcf = k1.ex_cw_dcf = pow(-1, t_last_1) * (convert_tensor(s_last_1) - convert_tensor(s_last_0) - v_a)

        return k0, k1


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

        Note:
            Distributed comparison function:
                f(x)=beta, if x < alpha; f(x)=0, else

        Args:
            num_of_keys (int): The number of keys to generate.
            alpha (RingTensor): The comparison point (private value for comparison).
            beta (RingTensor): The output value if the comparison is true.

        Returns:
            Tuple[DCFKey, DCFKey]: A tuple containing two DCFKey objects for the two parties.

        Examples:
            >>> k0, k1 = DCF.gen(10, alpha, beta)
        """
        return DCFKey.gen(num_of_keys, alpha, beta)

    @staticmethod
    def eval(x, keys, party_id, prg_type=PRG_TYPE):
        """
        Evaluate the DCF on input `x`.

        With this method, the party evaluates the function value locally using the input *x*,
        provided keys and party ID. It performs bitwise operations to securely
        compute the comparison result, which is the **shared value** of the result of the original function.

        Args:
            x (RingTensor): The input RingTensor on which the DCF is evaluated.
            keys (DCFKey): The secret sharing keys required for evaluation.
            party_id (int): The party ID (0 or 1), identifying which party is performing the evaluation.
            prg_type (str, optional): The type of pseudorandom generator (PRG) used during evaluation, defaults to PRG_TYPE.

        Returns:
            RingTensor: The sharing value of the comparison result for each party as a RIngTensor.

        Examples:
            >>> res = DCF.eval(x, key, 0)
        """
        shape = x.shape
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


class DICFKey(Parameter):
    """
    The secret sharing key for distributed interval comparison function (DICF).

    This class implements the secret sharing keys required for function secret sharing (FSS) in
    distributed interval comparison functions (DICF).
    The generation of the DICF key is based on the DCF key.
    It also includes methods for generating other key parameters like r_in and z used in the DICF protocol.

    Attributes:
        dcf_key (DCFKey): The key of DICF.
        r_in (RingTensor): The offset of the function, for the purpose of blind the input.
        z (RingTensor): The parameter for offline support.
    """

    def __init__(self):
        """
        Initialize the DICFKey object.

        This method initializes the key `dcf_key` to *DCFKey* object, the offset `r_in` to *None*,
        and the parameter `z` to *None*.

        Examples:
            >>> key = DICFKey()
        """
        self.dcf_key = DCFKey()
        self.r_in = None
        self.z = None
        self.size = 0

    @staticmethod
    def gen(num_of_keys, down_bound=RingTensor(0), upper_bound=RingTensor(HALF_RING - 1)):
        """
        Generate DICF keys.

        This method generates multiple DICF keys required for secure comparison.

        Note:
            Distributed point function:
                f(x)=1, if  down_bound< x < upper_bound; f(x)=0, else

        Args:
            num_of_keys (int): The number of keys to generate.
            down_bound (RingTensor): The down bound of the interval for DICF, default is 0.
            upper_bound (RingTensor): The upper bound of the interval for DICF, default is *HALF_RING-1*.

        Returns:
            Tuple[DICFKey, DICFKey]: A tuple containing two DICFKey objects for the two partys.

        Examples:
            >>> k0, k1 = DICFKey.gen(10, down, up)
        """
        upper_bound = upper_bound.tensor
        down_bound = down_bound.tensor

        r_in = RingTensor.random([num_of_keys, 1], dtype='int')
        gamma = r_in - 1

        r_tensor = r_in.tensor
        # 修正参数
        q1 = (upper_bound + 1)
        ap = (down_bound + r_tensor)
        aq = (upper_bound + r_tensor)
        aq1 = (upper_bound + 1 + r_tensor)

        out = ((ap > aq) + 0) - ((ap > down_bound) + 0) + ((aq1 > q1) + 0) + ((aq == -1) + 0)

        k0 = DICFKey()
        k1 = DICFKey()

        keys = DCFKey.gen(num_of_keys, gamma, RingTensor(1))

        k0.dcf_key, k1.dcf_key = keys

        z_share = RingTensor.random([num_of_keys], device=DEVICE)
        r_share = RingTensor.random([num_of_keys], device=DEVICE)

        k0.z, k1.z = out.squeeze(1) - z_share, z_share
        k0.r_in, k1.r_in = r_in.squeeze(1) - r_share, r_share

        return k0, k1


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

        Args:
            num_of_keys (int): The number of keys to generate.
            down_bound (RingTensor): The down bound of the interval for DICF, default is 0.
            upper_bound (RingTensor): The upper bound of the interval for DICF, default is *HALF_RING-1*.

        Returns:
            Tuple[DICFKey, DICFKey]: A tuple containing two DICFKey objects for the two parties.

        Examples:
            >>> k0, k1 = DICF.gen(10, down, up)
        """
        return DICFKey.gen(num_of_keys, down_bound, upper_bound)

    @staticmethod
    def eval(x_shift, keys, party_id, down_bound=RingTensor(0), upper_bound=RingTensor(HALF_RING - 1)):
        """
        Evaluate the DICF on input *x_shift*.

        With this method, the party evaluates the function value locally using the input *x_shift*.
        The main principle of this method is to implement DICF with DCF twice.

        Args:
            x_shift (RingTensor): The input RingTensor on which the DICF is evaluated.
            keys (DICFKey): The secret sharing keys required for evaluation.
            party_id (int): The party ID (0 or 1), identifying which party is performing the evaluation.
            down_bound (RingTensor): The down bound of the interval for DICF, default is 0.
            upper_bound (RingTensor): The upper bound of the interval for DICF, default is *HALF_RING-1*.

        Returns:
            RingTensor: The sharing value of the comparison result for each party as a RingTensor.

        Examples:
            >>> res = DICF.eval(x, key, 0)
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


class GrottoDICFKey(Parameter):
    """
    The secret sharing key for distributed interval comparison function (DICF) with the method adapted from `Grotto <https://eprint.iacr.org/2023/108>`_.

    This class implements the secret sharing keys required for function secret sharing (FSS) in
    distributed interval comparison functions (DICF).
    The generation of the GrottoDICF key is based on the DPF key.
    It also includes methods for generating other key parameters like r_in used in the GrottoDICF protocol.

    Attributes:
        dpf_key (DPFKey): The key of GrottoDICF.
        r_in (RingTensor): The offset of the function, for the purpose of blind the input.
    """

    def __init__(self):
        """
        Initialize the GrottoDICFKey object.

        This method initializes the key `dpf_key` to *DPFKey* object, the offset `r_in` to *None*.

        Examples:
            >>> key = GrottoDICFKey()
        """
        self.dpf_key = DPFKey()
        self.r_in = None

    @staticmethod
    def gen(num_of_keys, beta=RingTensor(1)):
        """
        Generate GrottoDICF keys.

        This method generates the DICF keys required for secure comparison with method adapted from Grotto.

        Args:
            num_of_keys (int): The number of keys to generate.
            beta (RingTensor): The output value if the comparison is true.

        Returns:
            Tuple[GrottoDICFKey, GrottoDICFKey]: A tuple containing two DICFKey objects for the two partys.

        Examples:
            >>> k0, k1 = GrottoDICFKey.gen(10, beta)
        """
        k0, k1 = GrottoDICFKey(), GrottoDICFKey()
        k0.r_in = RingTensor.random([num_of_keys])
        k1.r_in = RingTensor.random([num_of_keys])
        k0.dpf_key, k1.dpf_key = DPFKey.gen(num_of_keys, k0.r_in + k1.r_in, beta)
        return k0, k1


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

        Args:
            num_of_keys: The number of keys to generate.
            beta (RingTensor): The output value if the comparison is true.

        Returns:
            Tuple[GrottoDICFKey, GrottoDICFKey]: A tuple containing two GrottoDICFKey objects for the two parties.

        Examples:
            >>> k0, k1 = GrottoDICF.gen(10, beta)
        """
        return GrottoDICFKey.gen(num_of_keys, beta)

    @staticmethod
    def eval(x_shift: RingTensor, key, party_id, prg_type=PRG_TYPE, down_bound=RingTensor(0),
             upper_bound=RingTensor(HALF_RING - 1)):
        """
        Evaluate the DICF on input *x_shift* with the method adapted from Grotto.

        With this method, the party evaluates the function value locally using the input *x_shift*.

        Note:
            Emphasis on solving the hyper-ring situation. ans and tau.

        Args:
            x_shift: The input RingTensor on which the GrottoDICF is evaluated.
            key (GrottoDICFKey): The secret sharing keys required for evaluation.
            party_id (int): The party ID (0 or 1), identifying which party is performing the evaluation.
            prg_type (str, optional): The type of pseudorandom generator (PRG) used during evaluation, defaults to PRG_TYPE.
            down_bound (RingTensor): The down bound of the interval for GrottoDICF, default is 0.
            upper_bound (RingTensor): The upper bound of the interval for GrottoDICF, default is *HALF_RING-1*.

        Returns:
            RingTensor: The sharing value of the comparison result for each party as a RingTensor.

        Examples:
            >>> res = GrottoDICF.eval(x, key, 0)
        """
        p = down_bound + x_shift
        q = upper_bound + x_shift
        cond = (p ^ q) < 0
        tau = ((p > q) ^ cond) * party_id
        x = RingTensor.stack([p, q]).view(2, -1, 1)

        parity_x = prefix_parity_query(x, key.dpf_key, party_id, prg_type)

        ans = (parity_x[0] ^ parity_x[1]).view(x_shift.shape) ^ tau

        return ans


class SigmaDICFKey(Parameter):
    """
    The secret sharing key for distributed interval comparison function (DICF) with the method adapted from `Sigma <https://eprint.iacr.org/2023/1269>`_.

    This class implements the secret sharing keys required for function secret sharing (FSS) in
    distributed interval comparison functions (DICF).
    The generation of the SigmaDICF key is based on the DPF key.
    It also includes methods for generating other key parameters like c and r_in used in the SigmaDICF protocol.

    Attributes:
        dpf_key (DPFKey): The key of SigmaDICF.
        c (RingTensor): The parameter for offline support.
        r_in (RingTensor): The offset of the function, for the purpose of blind the input.
    """

    def __init__(self):
        """
        Initialize the SigmaDICFKey object.

        This method initializes the key `dpf_key` to *DPFKey* object, the MSB of offset `c` to *None*,
        and the offset `r_in` to *None*.

        Examples:
            >>> key = SigmaDICFKey()
        """
        self.dpf_key = DPFKey()
        self.c = None
        self.r_in = None
        self.size = 0

    @staticmethod
    def gen(num_of_keys, bit_len=BIT_LEN):
        """
        Generate SigmaDICF keys.

        This method generates the DICF keys required for secure comparison with method adapted from Sigma.

        Args:
            num_of_keys (int): The number of keys to generate.
            bit_len (int): The length of the binary bits of the offset `r_in`.

        Returns:
            Tuple[SigmaDICFKey, SigmaDICFKey]: A tuple containing two SigmaDICFKey objects for the two parties.

        Examples:
            >>> k0, k1 = SigmaDICFKey.gen(10, 32)
        """
        k0 = SigmaDICFKey()
        k1 = SigmaDICFKey()

        k0.r_in = RingTensor.random([num_of_keys], down_bound=-2 ** (bit_len - 1), upper_bound=2 ** (bit_len - 1) - 1)
        k1.r_in = RingTensor.random([num_of_keys], down_bound=-2 ** (bit_len - 1), upper_bound=2 ** (bit_len - 1) - 1)
        r_in = k0.r_in + k1.r_in
        if bit_len < BIT_LEN:
            r_in = RingTensor.where(r_in > 2 ** (bit_len - 1) - 1, r_in - 2 ** bit_len, r_in)
            r_in = RingTensor.where(r_in < -2 ** (bit_len - 1), r_in + 2 ** bit_len, r_in)
        r_in.bit_len = bit_len

        y1 = r_in % (2 ** (bit_len - 1) - 1)
        k0.dpf_key, k1.dpf_key = DPFKey.gen(num_of_keys, y1, RingTensor(1))
        c = r_in.signbit()
        c0 = RingTensor.random([num_of_keys], device=DEVICE)
        c1 = c ^ c0

        k0.c = c0
        k1.c = c1

        return k0, k1


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

        Args:
            num_of_keys (int): The number of keys to generate.

        Returns:
            Tuple[SigmaDICFKey, SigmaDICFKey]: A tuple containing two SigmaDICFKey objects for the two parties.

        Examples:
            >>> k0, k1 = SigmaDICF.gen(10)
        """
        return SigmaDICFKey.gen(num_of_keys)

    @staticmethod
    def eval(x_shift: RingTensor, key, party_id):
        """
        Evaluate the DICF on input *x_shift* with the method adapted from Sigma.

        With this method, the party evaluates the function value locally using the input *x_shift*.

        Args:
            x_shift: The input RingTensor on which the SigmaDICF is evaluated.
            key (SigmaDICFKey): The secret sharing keys required for evaluation.
            party_id (int): The party ID (0 or 1), identifying which party is performing the evaluation.

        Returns:
            RingTensor: The sharing value of the comparison result for each party as a RingTensor.

        Examples:
            >>> res = SigmaDICF.eval(x, key, 0)
        """
        shape = x_shift.shape
        x_shift = x_shift.contiguous().view(-1, 1)
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

        Args:
            input_list (list[RingTensor]): The input RingTensor on which the SigmaDICF is evaluated.
            key (SigmaDICFKey): The secret sharing keys required for evaluation.
            party_id (int): The party ID (0 or 1), identifying which party is performing the evaluation.

        Returns:
            Tuple: The tuple including the shared result of VSigma and the verification mark.

        Examples:
            >>> res, h = SigmaDICF.one_key_eval(x, key, 10, 0)
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


class VDPFKey(Parameter):
    """
    The FSS key class for verifiable distributed point function (VDPF).

    This class implements the generation method for the key used for VDPF evaluation.
    It includes methods for generating and managing the parameters used in VDPF protocol.

    Attributes:
        s (torch.Tensor): A binary string (lambda bits) generated by the PRG for the root node.
        cw_list (CWList): The list of correction words.
        ocw (torch.Tensor): The extra check word used for VDPF calculation.
        cs (torch.Tensor): The parameter required for verification.
    """

    def __init__(self):
        """
        Initialize the VDPFKey object.

        This method initializes the seed `s` to **None**, the list of correction words `cw_list` to a **CWList** object,
        the extra correction word `ocw` to **None**, and sets the verification parameter `cs` to **None**.

        Examples:
            >>> key = VDPFKey()
        """
        self.s = None
        self.cw_list = CWList()
        self.ocw = None
        self.cs = None

    @staticmethod
    def gen(num_of_keys, alpha, beta):
        """
        Generate keys for VDPF.

        This method can generate multiple keys for VDPF, which can be used for the following evaluation.

        Args:
            num_of_keys (int): Number of keys to generate.
            alpha (RingTensor): The comparison point of VDPF.
            beta (RingTensor): The output value if the comparison is true.

        Returns:
            tuple: The keys of all the involved parties (two parties).

        Examples:
            >>> k0, k1 = VDPFKey.gen(10, alpha, beta)
        """
        return vdpf_gen(num_of_keys, alpha, beta)


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

        Args:
            num_of_keys (int): Number of keys to generate.
            alpha (RingTensor): The comparison point of VDPF.
            beta (RingTensor): The output value if the comparison is true.

        Returns:
            tuple: The keys of all the involved parties (two parties).

        Examples:
            >>> k0, k1 = VDPF.gen(10, alpha, beta)
        """
        return VDPFKey.gen(num_of_keys, alpha, beta)

    @staticmethod
    def _eval(x: RingTensor, keys: VDPFKey, party_id):
        """
        Evaluate the output share for one party.

        According to the input x, the party calculates the function value locally,
        which is the shared value of the original VDPF.

        Args:
            x: The input to be evaluated.
            keys: The FSS key used for evaluation.
            party_id: The party ID (0 or 1), identifying which party is performing the evaluation.

        Returns:
            Tuple: The tuple including the shared result of VDPF and the verification mark.

        Examples:
            >>> res, h = VDPF._eval(x, key, 0)
        """
        prg = PRG(PRG_TYPE, DEVICE)

        t_last = party_id

        s_last = keys.s

        for i in range(x.bit_len):
            cw = keys.cw_list[i]

            s_cw = cw.s_cw
            t_cw_l = cw.t_cw_l
            t_cw_r = cw.t_cw_r

            s_l, v_l, t_l, s_r, v_r, t_r = CW.gen_dcf_cw(prg, s_last, LAMBDA)

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

    @staticmethod
    def eval(x, keys, party_id):
        """
        Evaluate the output share for one party.

        According to the input x, the party calculates the function value locally,
        which is the shared value of the original VDPF.
        And return the verification mark of the gen and eval process, thus ensuring the security.

        Args:
            x (RingTensor): The input to be evaluated.
            keys (VDPFKey): The FSS key used for evaluation.
            party_id (int): The party ID (0 or 1), identifying which party is performing the evaluation.

        Returns:
            Tuple: The tuple including the shared result of VDPF and the verification mark.

        Examples:
            >>> res, h = VDPF.eval(x, key, 0)
        """
        shape = x.shape
        x = x.view(-1, 1)
        res, h = VDPF._eval(x, keys, party_id)
        return res.reshape(shape), h

    @staticmethod
    def one_key_eval(x, keys, num, party_id):
        """
        Evaluate multiple inputs use only one VDPF key.

        According to the input x, the party calculates the function value locally,
        which is the shared value of the original VDPF.
        The emphasis is in evaluating multiple inputs use only one key.

        Args:
            x (RingTensor): The input to be evaluated.
            keys (VDPFKey): The FSS key used for evaluation.
            num (int): The number of input to be evaluated.
            party_id (int): The party ID (0 or 1), identifying which party is performing the evaluation.

        Returns:
            Tuple: The tuple including the shared result of VDPF and the verification mark.

        Examples:
            >>> res, h = VDPF.one_key_eval(x, key, 10, 0)
        """
        shape = x.shape
        x = x.view(num, -1, 1)
        res, h = VDPF._eval(x, keys, party_id)
        return res.reshape(shape), h

    @staticmethod
    def ppq(x, keys, party_id):
        """
        Implement the verifiable prefix parity query.

        This method implements a verifiable function on top of the original ppq method,
        ensuring the correctness of evaluation stage.
        For more information on ppq, please click `here <NssMPC.crypt.primitives.function_secret_sharing.dpf.prefix_parity_query>`.

        Args:
            x (RingTensor): The input to be evaluated.
            keys (VDPFKey): The FSS key used for evaluation.
            party_id (int): The party ID (0 or 1), identifying which party is performing the evaluation.

        Returns:
            Tuple: The tuple including the shared result of VDPF and the verification mark.

        Examples:
            >>> res, pi = VDPF.ppq(x, key, 0)
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

            s_l, t_l, s_r, t_r = gen_dpf_cw(prg._prg, s_last, LAMBDA)

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


class VSigmaKey(Parameter):
    """
    The FSS key class for verifiable sigma (VSigma).

    This class implements the generation method for the key used for VSigma evaluation.
    It includes methods for generating and managing the parameters used in VSigma protocol.

    Attributes:
        dpf_key (VDPFKey): The key of VSigma.
        c (RingTensor): The parameter for offline support.
        r_in (RingTensor): The offset of the function, for the purpose of blind the input.
    """

    def __init__(self, ver_dpf_key=VDPFKey()):
        """
        Initialize the VSigmaKey object.

        This method initializes the key `dpf_key` to *DPFKey* object, the MSB of offset `c` to *None*,
        and the offset `r_in` to *None*.

        Args:
            ver_dpf_key (VDPFKey, optional): The verifiable DPF key. Defaults to VDPFKey().

        Examples:
            >>> key = VSigmaKey()
        """
        self.ver_dpf_key = ver_dpf_key
        self.c = None
        self.r_in = None

    def __iter__(self):
        """
        Return an iterator that contains the three attributes of the class.

        This method allows you to access the three attributes of a class instance in sequence as it is iterated over.

        Returns:
            iterator: An iterator that contains the three attributes of the class.

        Examples:
            >>> r_in, dpf_key, c = list(key)
        """
        return iter([self.r_in, self.ver_dpf_key, self.c])

    @staticmethod
    def gen(num_of_keys):
        """
        Generate keys for VSigma.

        This method can generate multiple keys for VSigma, which can be used for the following evaluation.

        Args:
            num_of_keys (int): Number of keys to generate.

        Returns:
            tuple: The keys of all the involved parties (two parties).

        Examples:
            >>> k0, k1 = VSigmaKey.gen(10)
        """
        return verifiable_sigma_gen(num_of_keys)

    @classmethod
    def load_provider(cls, party: Party3PC):
        """
        Load the provider for VSigmaKey.

        Args:
            party (Party3PC): The party instance.

        Examples:
            >>> VSigmaKey.load_provider(party)
        """
        provider = ParamProvider(cls, saved_name=f'VSigmaKey_{(party.party_id + 1) % 3}_0')
        provider.load_param()
        party.virtual_party_with_previous.append_provider(provider)

        provider = ParamProvider(cls, saved_name=f'VSigmaKey_{(party.party_id - 1) % 3}_1')
        provider.load_param()
        party.virtual_party_with_next.append_provider(provider)

        provider = ParamProvider(cls, saved_name=f'VSigmaKey_{party.party_id}_0',
                                 param_tag=f'VSigmaKey_{party.party_id}_0')
        provider.load_param()
        party.append_provider(provider)

        provider = ParamProvider(cls, saved_name=f'VSigmaKey_{party.party_id}_1',
                                 param_tag=f'VSigmaKey_{party.party_id}_1')
        provider.load_param()
        party.append_provider(provider)


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

        Args:
            num_of_keys (int): Number of keys to generate.

        Returns:
            tuple: The keys of all the involved parties (two parties).

        Examples:
            >>> k0, k1 = VSigma.gen(10)
        """
        return VSigmaKey.gen(num_of_keys)

    @staticmethod
    def eval(x_shift, keys, party_id):
        """
        Evaluate the output share for one party.

        According to the input x_shift, the party calculates the function value locally,
        which is the shared value of the original VSigma.
        And return the verification mark of the gen and eval process, thus ensuring the security.

        Args:
            x_shift (RingTensor): The input to be evaluated.
            keys (VSigmaKey or tuple): The FSS key used for evaluation.
            party_id (int): The party ID (0 or 1), identifying which party is performing the evaluation.

        Returns:
            Tuple: The tuple including the shared result of VSigma and the verification mark.

        Examples:
            >>> res, pi = VSigma.eval(x, key, 0)
        """
        return verifiable_sigma_eval(party_id, keys, x_shift)

    @staticmethod
    def cmp_eval(x, keys, party_id):
        """
        Evaluate the comparison result for one party.

        Args:
            x (RingTensor): The input to be evaluated.
            keys (VSigmaKey): The FSS key used for evaluation.
            party_id (int): The party ID (0 or 1), identifying which party is performing the evaluation.

        Returns:
            Tuple: The tuple including the shared result of VSigma and the verification mark.

        Examples:
            >>> res, pi = VSigma.cmp_eval(x, key, 0)
        """
        if DEBUG_LEVEL == 2:
            x_shift = AdditiveSecretSharing(keys.r_in) + x
        else:
            x_shift = AdditiveSecretSharing(keys.r_in.reshape(x.shape)) + x
        x_shift = x_shift.restore()
        return verifiable_sigma_eval(party_id, keys, x_shift)


def verifiable_sigma_eval(party_id, key, x_shift):
    """
    Evaluate the output share for one party.

    According to the input x_shift, the party calculates the function value locally,
    which is the shared value of the original VSigma.
    And return the verification mark of the gen and eval process, thus ensuring the security.

    Args:
        party_id (int): The party ID (0 or 1), identifying which party is performing the evaluation.
        key (VSigmaKey or tuple): The FSS key used for evaluation.
        x_shift (RingTensor): The input to be evaluated.

    Returns:
        Tuple: The tuple including the shared result of VSigma and the verification mark.

    Examples:
        >>> res, pi = verifiable_sigma_eval(0, key, x)
    """
    shape = x_shift.shape
    x_shift = x_shift.reshape(-1, 1)
    K, c = key
    y = x_shift % (HALF_RING - 1)
    y = y + 1
    out, pi = VDPF.ppq(y, K, party_id)
    out = x_shift.signbit() * party_id ^ c.reshape(-1, 1) ^ out
    return out.reshape(shape), pi


@torch.jit.script
def gen_dpf_cw(prg: torch.classes.csprng_aes.AES_PRG, new_seeds, lmd: int, bit_len: int = BIT_LEN):
    """
    Generate the Correction Words (CW) for the Distributed Point Function (DPF).

    Similar to DCF, this method generates the correction words for DPF, which allows evaluating equality checks (:math:`x = \\alpha`) securely.
    It handles the path verification and value correction at the specified tree level.

    Note:
        Only PRG with AES kernel is supported.

    Args:
        prg: The AES-based pseudorandom generator instance.
        new_seeds (RingTensor or torch.Tensor): The input seeds for the current layer.
        lmd: The security parameter lambda.
        bit_len: The domain size (bit length) of the input. Defaults to global BIT_LEN.

    Returns:
        tuple: A tuple containing the components of the Correction Words.

    Raises:
        ValueError: If the PRG kernel is not 'AES'.

    Examples:
        >>> s_l, t_l, s_r, t_r = gen_dpf_cw(prg, seeds, 128)
    """
    prg.set_seeds(new_seeds)

    random_bits = prg.bit_random(2 * lmd + 2)
    s_num = 128 // bit_len
    s_l_res = random_bits[..., 0:s_num]

    s_r_res = random_bits[..., s_num: s_num + s_num]

    t_l_res = random_bits[..., s_num + s_num] & 1
    t_l_res = t_l_res.unsqueeze(-1)
    t_r_res = random_bits[..., s_num + s_num] >> 1 & 1
    t_r_res = t_r_res.unsqueeze(-1)
    return s_l_res, t_l_res, s_r_res, t_r_res


def vdpf_gen(num_of_keys, alpha, beta):
    """
    Generate keys for VDPF.

    This method can generate multiple keys for VDPF, which can be used for the following evaluation.

    Args:
        num_of_keys (int): Number of keys to generate.
        alpha (RingTensor): The comparison point of VDPF.
        beta (RingTensor): The output value if the comparison is true.

    Returns:
        tuple: The keys of all the involved parties (two parties)

    Examples:
        >>> k0, k1 = vdpf_gen(10, alpha, beta)
    """
    seed_0 = torch.randint(-HALF_RING, HALF_RING - 1, [num_of_keys, LAMBDA // BIT_LEN], dtype=data_type, device=DEVICE)
    seed_1 = torch.randint(-HALF_RING, HALF_RING - 1, [num_of_keys, LAMBDA // BIT_LEN], dtype=data_type, device=DEVICE)
    # 产生伪随机数产生器的种子

    prg = PRG(PRG_TYPE, device=DEVICE)
    prg.set_seeds(seed_0)
    s_0_0 = prg.bit_random_tensor(LAMBDA)
    prg.set_seeds(seed_1)
    s_0_1 = prg.bit_random_tensor(LAMBDA)

    k0 = VDPFKey()
    k1 = VDPFKey()

    k0.s = s_0_0
    k1.s = s_0_1

    s_last_0 = s_0_0
    s_last_1 = s_0_1

    t_last_0 = 0
    t_last_1 = 1
    prg = PRG(PRG_TYPE, DEVICE)

    for i in range(alpha.bit_len):
        s_l_0, t_l_0, s_r_0, t_r_0 = gen_dpf_cw(prg._prg, s_last_0, LAMBDA)
        s_l_1, t_l_1, s_r_1, t_r_1 = gen_dpf_cw(prg._prg, s_last_1, LAMBDA)

        cond = (alpha.get_tensor_bit(alpha.bit_len - 1 - i) == 0).view(-1, 1)

        l_tensors = [s_l_0, s_l_1, t_l_0, t_l_1]
        r_tensors = [s_r_0, s_r_1, t_r_0, t_r_1]

        keep_tensors = [torch.where(cond, l, r) for l, r in zip(l_tensors, r_tensors)]
        lose_tensors = [torch.where(cond, r, l) for l, r in zip(l_tensors, r_tensors)]

        s_keep_0, s_keep_1, t_keep_0, t_keep_1 = keep_tensors
        s_lose_0, s_lose_1, t_lose_0, t_lose_1 = lose_tensors

        s_cw = s_lose_0 ^ s_lose_1

        t_l_cw = t_l_0 ^ t_l_1 ^ ~cond ^ 1
        t_r_cw = t_r_0 ^ t_r_1 ^ ~cond

        cw = CW(s_cw=s_cw, t_cw_l=t_l_cw, t_cw_r=t_r_cw, lmd=LAMBDA)

        k0.cw_list.append(cw)
        k1.cw_list.append(cw)

        t_keep_cw = torch.where(cond, t_l_cw, t_r_cw)

        s_last_0 = s_keep_0 ^ (t_last_0 * s_cw)
        s_last_1 = s_keep_1 ^ (t_last_1 * s_cw)

        t_last_0 = t_keep_0 ^ (t_last_0 * t_keep_cw)
        t_last_1 = t_keep_1 ^ (t_last_1 * t_keep_cw)

    # TODO: Hash function
    # prg.set_seeds(torch.cat((s_last_0, alpha.tensor.unsqueeze(1)), dim=1))
    prg.set_seeds(s_last_0 + alpha.tensor.unsqueeze(1))
    pi_0 = prg.bit_random_tensor(4 * LAMBDA)
    # prg.set_seeds(torch.cat((s_last_1, alpha.tensor.unsqueeze(1)), dim=1))
    prg.set_seeds(s_last_1 + alpha.tensor.unsqueeze(1))
    pi_1 = prg.bit_random_tensor(4 * LAMBDA)

    s_0_n_add_1 = s_last_0
    s_1_n_add_1 = s_last_1

    # t_0_n_add_1 = s_0_n_add_1 & 1
    # t_1_n_add_1 = s_1_n_add_1 & 1
    cs = pi_0 ^ pi_1
    k0.cs = k1.cs = cs
    k0.ocw = k1.ocw = pow(-1, t_last_1) * (
            beta.tensor - convert_tensor(s_0_n_add_1) + convert_tensor(s_1_n_add_1))

    return k0, k1


def verifiable_sigma_gen(num_of_keys):
    """
    Generate keys for VSigma.

    This method can generate multiple keys for VSigma, which can be used for the following evaluation.

    Args:
        num_of_keys (int): Number of keys to generate.

    Returns:
        tuple: The keys of all the involved parties (two parties).

    Examples:
        >>> k0, k1 = verifiable_sigma_gen(10)
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

    k0.r_in, k1.r_in = AdditiveSecretSharing.share(r_in, 2)

    return k0, k1
