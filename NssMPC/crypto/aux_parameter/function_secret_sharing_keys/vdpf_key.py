import torch

from NssMPC.common.random import PRG
from NssMPC.common.utils import convert_tensor
from NssMPC.config import HALF_RING, LAMBDA, data_type, BIT_LEN, DEVICE, PRG_TYPE
from NssMPC.crypto.aux_parameter import Parameter
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys import CW, CWList


class VDPFKey(Parameter):

    def __init__(self):
        self.s = None
        self.cw_list = CWList()
        self.ocw = None
        self.cs = None

    @staticmethod
    def gen(num_of_keys, alpha, beta):
        """
        分布式点函数密钥生成接口
        通过该接口实现多个密钥生成
        分布式点函数：
        f(x)=b, if x = α; f(x)=0, else

        :param num_of_keys: 需要的密钥数量
        :param alpha: 分布式比较函数的参数α
        :param beta: 分布式比较函数参数b
        :return: 各个参与方（两方）的密钥(tuple)
        """
        return vdpf_gen(num_of_keys, alpha, beta)


def vdpf_gen(num_of_keys, alpha, beta):
    """
    通过伪随机数生成器并行产生各参与方的dpf密钥
    :param num_of_keys: 所需密钥数量
    :param alpha: 分布式点函数的参数α
    :param beta: 分布式点函数的参数b
    :return: 各参与方的密钥
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
        s_l_0, t_l_0, s_r_0, t_r_0 = CW.gen_dpf_cw(prg, s_last_0, LAMBDA)
        s_l_1, t_l_1, s_r_1, t_r_1 = CW.gen_dpf_cw(prg, s_last_1, LAMBDA)

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
