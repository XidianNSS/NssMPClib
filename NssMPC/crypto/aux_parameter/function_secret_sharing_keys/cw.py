import torch

from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.config import BIT_LEN, data_type


class CW(object):
    """Correction Words, CW
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __setitem__(self, key, value):
        for k, v in self.__dict__.items():
            if hasattr(v, '__setitem__'):
                v[key] = getattr(value, k)

    def __getitem__(self, item):
        ret = CW()
        for k, v in self.__dict__.items():
            if hasattr(v, '__getitem__'):
                setattr(ret, k, v[item])
        return ret

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, (RingTensor, torch.Tensor)) or hasattr(v, 'to'):
                setattr(self, k, v.to(device))
        return self

    @staticmethod
    def gen_dcf_cw(prg, new_seeds, lmd):
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
    def __init__(self, *args):
        super().__init__(args)

    def getitem(self, item):
        ret = CWList()
        for element in self:
            ret.append(element[item])
        return ret

    def setitem(self, item, value):
        for i in range(len(self)):
            self[i][item] = value[i]

    def to(self, device):
        temp = CWList()
        for v in self:
            if isinstance(v, CW) or isinstance(v, (RingTensor, torch.Tensor)) or hasattr(v, 'to'):
                temp.append(v.to(device))
        return temp

    def expand_as(self, input):
        ret = CWList()
        for i, value in enumerate(self):
            if hasattr(value, 'expand_as'):
                ret.append(value.expand_as(input[i]))
        return ret
