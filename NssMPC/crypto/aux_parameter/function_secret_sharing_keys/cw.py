#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch

from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.config import BIT_LEN


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

        :param kwargs: The attributes that need to be set.
        :type kwargs: keyword, in a more specific form: str=any, for example: attr=[1, 2, 3]
        """
        self.__dict__.update(kwargs)

    def __setitem__(self, key, value):
        """
        Assign values to attributes of CW instances by index or key.

        Iterate over all the properties of the object, and if they support an index or key assignment,
        they are dynamically updated based on the key and value (the CW instance that contains the value to assign to) passed in.

        :param key: The key or index of each property to assign to.
        :type key: The type supported by the index or key
        :param value: The CW instance containing the value of the property to be assigned.
        :type value: CW
        :return: The CW instance after updating the property value.
        :rtype: CW
        """
        for k, v in self.__dict__.items():
            if hasattr(v, '__setitem__'):
                v[key] = getattr(value, k)

    def __getitem__(self, item):
        """
        Get the value in the CW object by index or key.

        Iterate over all the properties of the object, try to extract values from those that support index or key assignment,
        and store them in a new CW object and return it.

        :param item: The key or index you want to use to retrieve the value.
        :type item: The type supported by the index or key
        :return: The CW instance after obtaining the corresponding property value.
        :rtype: CW
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

        :param device: The target device to transfer to.
        :type device: str
        :return: The CW object after transferring the device.
        :rtype: CW
        """
        for k, v in self.__dict__.items():
            if isinstance(v, (RingTensor, torch.Tensor)) or hasattr(v, 'to'):
                setattr(self, k, v.to(device))
        return self

    @staticmethod
    def gen_dcf_cw(prg, new_seeds, lmd):
        """
        Generate the CW (correction words) for DCF.

        We use this method to generate the correction words used for the computation of DCF,
        and return the items contained in the correction words as tuples.

        .. note::
            Only PRG with AES kernel is supported.

        :param prg: The pseudorandom generator (prg) we used for the generation of cw.
        :type prg: PRG
        :param new_seeds: The seeds used to set the prg.
        :type new_seeds: RingTensor or torch.Tensor
        :param lmd: The auxiliary parameter used to set the binary bit length of CW.
        :type lmd: int
        :return: The tuples containing the items of CW.
        :rtype: tuple
        :raises ValueError: If kernel is not 'AES'.
        """
        prg.set_seeds(new_seeds)
        if prg.kernel == 'AES':
            random_bits = prg.bit_random_tensor(4 * lmd + 2)
            s_num = 128 // BIT_LEN
            s_l_res = random_bits[..., 0:s_num]
            v_l_res = random_bits[..., s_num: s_num + s_num]

            s_r_res = random_bits[..., 2 * s_num: 2 * s_num + s_num]

            v_r_res = random_bits[..., 3 * s_num: 3 * s_num + s_num]

            t_l_res = random_bits[..., 4 * s_num] & 1
            t_l_res = t_l_res.unsqueeze(-1)
            t_r_res = random_bits[..., 4 * s_num] >> 1 & 1
            t_r_res = t_r_res.unsqueeze(-1)
            return s_l_res, v_l_res, t_l_res, s_r_res, v_r_res, t_r_res
        else:
            raise ValueError("kernel is not supported!")

    @staticmethod
    def gen_dpf_cw(prg, new_seeds, lmd):
        """
        Generate the CW (correction words) for DPF.

        We use this method to generate the correction words used for the computation of DPF,
        and return the items contained in the correction words as tuples.

        .. note::
            Only PRG with AES kernel is supported.

        :param prg: The pseudorandom generator (prg) we used for the generation of cw.
        :type prg: PRG
        :param new_seeds: The seeds used to set the prg.
        :type new_seeds: RingTensor or torch.Tensor
        :param lmd: The auxiliary parameter used to set the binary bit length of CW.
        :return: The tuples containing the items of CW.
        :rtype: tuple
        :raises ValueError: If kernel is not 'AES'.
        """
        prg.set_seeds(new_seeds)
        if prg.kernel == 'AES':
            random_bits = prg.bit_random_tensor(2 * lmd + 2)
            s_num = 128 // BIT_LEN
            s_l_res = random_bits[..., 0:s_num]

            s_r_res = random_bits[..., s_num: s_num + s_num]

            t_l_res = random_bits[..., s_num + s_num] & 1
            t_l_res = t_l_res.unsqueeze(-1)
            t_r_res = random_bits[..., s_num + s_num] >> 1 & 1
            t_r_res = t_r_res.unsqueeze(-1)
            return s_l_res, t_l_res, s_r_res, t_r_res
        else:
            raise ValueError("kernel is not supported!")


class CWList(list):
    """
    A list-like collection for CW instances used in FSS (Function Secret Sharing) computations.

    This class is used to aggregate the CW instances involved in FSS computations and defines
    corresponding operations on each CW in the list to reduce the complexity of the code during usage.
    """

    def __init__(self, *args):
        """Initialize a CWList instance"""
        super().__init__(args)

    def getitem(self, item):
        """
        Retrieve the value corresponding to the item from each element in the list.

        This method iterates over each element (CW instance) in the CWList,
        calls the indexing operation on these elements, and stores the result in a new CWList instance to return.
        It is equivalent to calling the __getitem__ method on each CW instance.

        :param item: The key or index you want to use to retrieve the value.
        :type item: The type supported by the index or key.
        :return: A new CWList instance containing the corresponding values from each CW instance.
        :rtype: CWList
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

        :param item: The key or index you want to use to assign the value.
        :type item: The type supported by the index or key.
        :param value: The CWList instance containing the value of the property to be assigned.
        :type value: CWList
        :return: The CWList object after transferring the instances to the specified device.
        :rtype: CWList
        """
        for i in range(len(self)):
            self[i][item] = value[i]

    def to(self, device):
        """
        Transfers the CW instance inside the CWList object that support the `to` operation to the specified device(`CPU` or `GPU`).

        This method iterates over all items (suppose to be CW instances) of the object, and if a item supports the `to` method,
        it is transferred to the target device.

        :param device: The target device to transfer to.
        :type device: str
        :return: The CWList object after transferring the device.
        :rtype: CWList
        """
        temp = CWList()
        for v in self:
            if isinstance(v, CW) or isinstance(v, (RingTensor, torch.Tensor)) or hasattr(v, 'to'):
                temp.append(v.to(device))
        return temp

    def expand_as(self, input):
        """
        Expands each element in CWList to be the same shape as the corresponding element in another input sequence.

        :param input: input sequence
        :type input: list or torch.Tensor
        :return: A new CWList instance containing the extended element
        :rtype: CWList
        """
        ret = CWList()
        for i, value in enumerate(self):
            if hasattr(value, 'expand_as'):
                ret.append(value.expand_as(input[i]))
        return ret
