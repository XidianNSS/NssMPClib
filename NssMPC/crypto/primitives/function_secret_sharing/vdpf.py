"""
本文件定义了安全两方计算情况下分布式点函数的函数秘密共享
在进行分布式点函数密钥产生和求值的过程中包含了分布式点函数的相关过程
本文件中的方法定义参考E. Boyle e.t.c. Function Secret Sharing: Improvements and Extensions.2016
https://dl.acm.org/doi/10.1145/2976749.2978429
"""

from NssMPC import RingTensor
from NssMPC.common.random import PRG
from NssMPC.common.utils import convert_tensor
from NssMPC.config import DEVICE, LAMBDA, PRG_TYPE
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys import CW
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.vdpf_key import VDPFKey


class VDPF(object):
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
        return VDPFKey.gen(num_of_keys, alpha, beta)

    @staticmethod
    def eval(x, keys, party_id):
        """
        分布式点函数EVAL过程接口
        根据输入x，参与方在本地计算函数值，即原函数f(x)的分享值

        :param x: 输入变量值x
        :param keys: 参与方关于函数分享的密钥
        :param party_id: 参与方编号
        :return: 分布式点函数的结果
        """
        shape = x.shape
        x = x.view(-1, 1)
        res, h = vdpf_eval(x, keys, party_id)
        return res.reshape(shape), h

    @staticmethod
    def one_key_eval(x, keys, num, party_id):
        """
        分布式点函数EVAL过程接口
        根据输入x，参与方在本地计算函数值，即原函数f(x)的分享值

        :param x: 输入变量值x
        :param keys: 参与方关于函数分享的密钥
        :param party_id: 参与方编号
        :return: 分布式点函数的结果
        """
        shape = x.shape
        x = x.view(num, -1, 1)
        res, h = vdpf_eval(x, keys, party_id)
        return res.reshape(shape), h

    @staticmethod
    def ppq(x, keys, party_id):
        return ver_ppq_dpf(x, keys, party_id)


def vdpf_eval(x: RingTensor, keys: VDPFKey, party_id):
    """
    分布式点函数EVAL过程
    根据输入x，参与方在本地计算函数值，即原函数f(x)的分享值

    :param x: 输入变量值x
    :param keys: 参与方关于函数分享的密钥
    :param party_id: 参与方编号
    :return: 分布式点函数的结果
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
