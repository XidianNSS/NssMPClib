from common.random.prg import PRG
from common.tensor import RingTensor
from config.base_configs import PRG_TYPE, DEVICE, LAMBDA
from crypto.primitives.function_secret_sharing.dpf import gen_dpf_cw


def prefix_parity_query(x: RingTensor, keys, party_id, prg_type=PRG_TYPE):
    """
    By transforming the distributed point function EVAL process
    to compute the prefix parity sum of a section (Prefix Parity Sum)

    Based on the input x, the participant locally computes the parity of the point in the construction tree

    Args:
        x: the input
        keys: the key of distributed point function
        party_id: the id of the party
        prg_type: the type of pseudorandom number generator

    Returns:
        the result of the prefix parity sum
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

        s_l, t_l, s_r, t_r = gen_dpf_cw(prg, s_last, LAMBDA)

        s1_l = s_l ^ (s_cw * t_last)
        t1_l = t_l ^ (t_cw_l * t_last)
        s1_r = s_r ^ (s_cw * t_last)
        t1_r = t_r ^ (t_cw_r * t_last)

        x_shift_bit = x.get_bit(x.bit_len - 1 - i)

        cond = (d != x_shift_bit)
        d = x_shift_bit * cond + d * ~cond

        psg_b = (psg_b ^ t_last) * cond + psg_b * ~cond

        s_last = s1_r * x_shift_bit + s1_l * (1 - x_shift_bit)
        t_last = t1_r * x_shift_bit + t1_l * (1 - x_shift_bit)

    psg_b = (psg_b ^ t_last) * d + psg_b * (1 - d)

    return psg_b
