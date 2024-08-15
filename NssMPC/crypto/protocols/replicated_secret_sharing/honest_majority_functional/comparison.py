import math

from NssMPC import RingTensor
from NssMPC.common.utils.common_utils import list_rotate
from NssMPC.config import BIT_LEN, HALF_RING, VCMP_SPLIT_LEN
from NssMPC.config.runtime import MAC_BUFFER
from NssMPC.crypto.aux_parameter import Parameter, DPFKey
from NssMPC.crypto.primitives.function_secret_sharing.dpf import DPF
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional import zero_encoding, one_encoding, b2a
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import coin, check_zero, open
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.multiplication import v_mul
from NssMPC.crypto.protocols.replicated_secret_sharing.semi_honest_functional import rand_like, mul_with_out_trunc


def secure_ge(x, y):  # TODO mac
    """
    Gets the most significant bit of an RSS sharing ⟨x⟩
    :param x: an RSS sharing ⟨x⟩
    :param y: an RSS sharing ⟨y⟩
    :return: the most significant bit of x
    """
    z = x - y
    dtype = x.dtype
    x.dtype = y.dtype = 'int'

    shape = z.shape

    next_party = z.party.virtual_party_with_next
    pre_party = z.party.virtual_party_with_previous

    key_from_next = next_party.get_param(MaliciousCMPKey, z.numel())
    key_from_previous = pre_party.get_param(MaliciousCMPKey, z.numel())

    x_next = x.item[0] + x.item[1]
    x_pre = x.item[1]
    y_next = y.item[0] + y.item[1]
    y_pre = y.item[1]

    from NssMPC import ArithmeticSecretSharing
    input_next = ArithmeticSecretSharing(zero_encoding(x_next - y_next + HALF_RING)[0], next_party)
    input_pre = ArithmeticSecretSharing(one_encoding(y_pre - x_pre + HALF_RING)[0], pre_party)

    out_pre, out_next = equal_match_all_dpf(ArithmeticSecretSharing(RingTensor(0), pre_party), input_pre, input_next,
                                            ArithmeticSecretSharing(RingTensor(0), pre_party),
                                            key_from_previous, key_from_next, VCMP_SPLIT_LEN)

    share_table, mac_key, mac_table = x.party.get_param(MACKey, z.numel())

    next = ArithmeticSecretSharing(out_next, next_party)
    pre = ArithmeticSecretSharing(out_pre, pre_party)

    v_next = share_table.item[1][..., 0].unsqueeze(-1) * next + share_table.item[1][..., 1].unsqueeze(-1) * (1 - next)
    v_pre = share_table.item[0][..., 0].unsqueeze(-1) * pre + share_table.item[0][..., 1].unsqueeze(-1) * (1 - pre)

    mac_v0 = mac_table.item[1][..., 0].unsqueeze(-1) * next + mac_table.item[1][..., 1].unsqueeze(-1) * (1 - next)
    mac_v1 = mac_table.item[0][..., 0].unsqueeze(-1) * pre + mac_table.item[0][..., 1].unsqueeze(-1) * (1 - pre)

    res = v_next + v_pre
    mac_v = mac_v0 + mac_v1
    res = x.__class__.reshare(res.item, x.party)
    mac_v = x.__class__.reshare(mac_v.item, x.party)
    mac_table.party = x.party

    MAC_BUFFER.add(res, mac_v, mac_key)
    # mac_check(res, mac_v, mac_key)

    res.dtype = dtype
    return res.view(shape) * res.scale


def v_bit_injection(x):
    """
    Bit injection of an RSS sharing ⟨x⟩B
    :param x: an RSS binary sharing ⟨x⟩B
    :return: an RSS arithmetic sharing ⟨x⟩A
    """
    zeros = RingTensor.zeros(x.shape, x.dtype, x.device)
    r_list = [x.__class__([x.item[0], zeros], x.party),
              x.__class__([zeros, x.item[1]], x.party),
              x.__class__([zeros, zeros], x.party)]

    r_list = list_rotate(r_list, x.party.party_id)
    a_x1, a_x2, a_x3 = r_list

    mul1 = v_mul(a_x1, a_x2)
    d = a_x1 + a_x2 - mul1 - mul1
    mul2 = v_mul(d, a_x3)
    b = d + a_x3 - mul2 - mul2

    return b


def mac_check(x, mx, mac_key):
    r = rand_like(x, x.party)
    mr = mul_with_out_trunc(r, mac_key)
    ro = coin(x.numel(), x.party).reshape(x.shape)
    v = r + x * ro
    w = mr + mx * ro
    v = open(v)
    check_zero(w - mac_key * v)


class MACKey(Parameter):
    def __init__(self, share_table=None, mac_key=None, smac_table=None):
        self.share_table = share_table
        self.mac_key = mac_key
        self.smac_table = smac_table

    def __iter__(self):
        return iter([self.share_table, self.mac_key, self.smac_table])

    @staticmethod
    def gen(num_of_keys):
        table_01 = RingTensor.zeros([num_of_keys, 2])
        table_01[:, 1] += 1

        from NssMPC import ReplicatedSecretSharing
        share_tables = ReplicatedSecretSharing.share(table_01)

        mac_key = RingTensor.random([num_of_keys, 1])
        share_keys = ReplicatedSecretSharing.share(mac_key)

        mac_table = table_01 * mac_key
        smac_tables = ReplicatedSecretSharing.share(mac_table)

        return [MACKey(a, b, c) for a, b, c in zip(share_tables, share_keys, smac_tables)]


class MaliciousCMPKey(Parameter):
    def __init__(self, k0=DPFKey(), r0=None, k1=DPFKey(), r1=None, k2=DPFKey(), r2=None):
        self.r0 = r0
        self.k0 = k0
        self.r1 = r1
        self.k1 = k1
        self.r2 = r2
        self.k2 = k2

    def __iter__(self):
        return iter((self.k0, self.r0, self.k1, self.r1, self.k2, self.r2))

    def __getitem__(self, item):
        ret = self.__class__()
        for attr, value in self.__dict__.items():
            if attr in ['k0', 'k1']:
                if attr == 'k0':
                    idx = BIT_LEN * BIT_LEN // VCMP_SPLIT_LEN
                else:
                    idx = BIT_LEN
                if isinstance(item, slice):
                    s_item = slice(idx * item.start, idx * item.stop, None)
                    setattr(ret, attr, value[s_item])
                else:
                    setattr(ret, attr, value[idx * item:idx * item + idx])
            else:
                setattr(ret, attr, value[item])

        return ret

    @staticmethod
    def gen(num_of_keys, bit_len):
        total_keys = num_of_keys * BIT_LEN * BIT_LEN // bit_len

        r0 = RingTensor.random([num_of_keys, BIT_LEN])

        split_r0 = _split_num(r0, bit_len).flatten()

        split_r0.bit_len = bit_len
        from NssMPC import ArithmeticSecretSharing
        r00, r01 = ArithmeticSecretSharing.share(r0, 2)
        r00.bit_len = r01.bit_len = bit_len
        k00, k01 = DPF.gen(total_keys, split_r0, RingTensor.convert_to_ring(1))

        r1 = RingTensor.random([num_of_keys, BIT_LEN], down_bound=0, upper_bound=BIT_LEN // bit_len)
        r1.bit_len = int(math.log2(BIT_LEN // bit_len)) + 1
        r10, r11 = ArithmeticSecretSharing.share(r1, 2)
        k10, k11 = DPF.gen(num_of_keys * BIT_LEN, r1, RingTensor.convert_to_ring(1))

        r2 = RingTensor.random([num_of_keys, 1], down_bound=0, upper_bound=BIT_LEN)
        r2.bit_len = int(math.log2(BIT_LEN))
        r20, r21 = ArithmeticSecretSharing.share(r2)
        k20, k21 = DPF.gen(num_of_keys, r2, RingTensor.convert_to_ring(1))

        return MaliciousCMPKey(k00, r00, k10, r10, k20, r20), MaliciousCMPKey(k01, r01, k11, r11, k21, r21)


def equal_match_key_gen_all_dpf(num_of_keys, bit_len):
    return MaliciousCMPKey.gen(num_of_keys, bit_len)


def equal_match_all_dpf(xp, yp, xn, yn, kp, kn, bit_len):
    kp0, rp0, kp1, rp1, kp2, rp2 = kp
    kn0, rn0, kn1, rn1, kn2, rn2 = kn

    party_p = rp1.party = rp2.party = xp.party
    party_n = rn1.party = rn2.party = xn.party

    delta_p = xp - yp
    delta_n = xn - yn

    xp_shift_shared = delta_p + rp0
    xn_shift_shared = delta_n + rn0

    if party_p.real_party.party_id == 2:
        xp_shift = xp_shift_shared.restore()
        xn_shift = xn_shift_shared.restore()
    else:
        xn_shift = xn_shift_shared.restore()
        xp_shift = xp_shift_shared.restore()

    split_xp = _split_num(xp_shift, bit_len)
    split_xn = _split_num(xn_shift, bit_len)

    split_xn.bit_len = split_xp.bit_len = bit_len

    res1_p = DPF.eval(split_xp.view(-1, len(kp0), 1), kp0, party_p.party_id)
    res1_p = res1_p.view(split_xp.size()).sum(dim=-1)
    res1_n = DPF.eval(split_xn.view(-1, len(kn0), 1), kn0, party_n.party_id)
    res1_n = res1_n.view(split_xn.size()).sum(dim=-1)

    from NssMPC import ArithmeticSecretSharing
    res1_p_shift_shared = rp1 + ArithmeticSecretSharing(res1_p, party_p)
    res1_n_shift_shared = rn1 + ArithmeticSecretSharing(res1_n, party_n)

    res1_p_shift_shared -= RingTensor(BIT_LEN // bit_len)
    res1_n_shift_shared -= RingTensor(BIT_LEN // bit_len)

    if party_p.real_party.party_id == 2:
        res1_p_shift = res1_p_shift_shared.restore()
        res1_n_shift = res1_n_shift_shared.restore()
    else:
        res1_n_shift = res1_n_shift_shared.restore()
        res1_p_shift = res1_p_shift_shared.restore()

    res1_p_shift.bit_len = len(kp1.cw_list)
    res1_n_shift.bit_len = len(kn1.cw_list)

    res2_p = DPF.eval(res1_p_shift.view(-1, len(kp1), 1), kp1, party_p.party_id)
    res2_p = res2_p.view(res1_p_shift.size()).sum(-1).view(-1,1)

    res2_n = DPF.eval(res1_n_shift.view(-1, len(kn1), 1), kn1, party_n.party_id)
    res2_n = res2_n.view(res1_n_shift.size()).sum(-1).view(-1,1)

    res2_p_shift_shared = rp2 + ArithmeticSecretSharing(res2_p, party_p)
    res2_n_shift_shared = rn2 + ArithmeticSecretSharing(res2_n, party_n)
    res2_p_shift_shared -= RingTensor.convert_to_ring(1)
    res2_n_shift_shared -= RingTensor.convert_to_ring(1)

    if party_p.real_party.party_id == 2:
        res2_p_shift = res2_p_shift_shared.restore()
        res2_n_shift = res2_n_shift_shared.restore()
    else:
        res2_n_shift = res2_n_shift_shared.restore()
        res2_p_shift = res2_p_shift_shared.restore()

    res2_p_shift.bit_len = len(kp2.cw_list)
    res2_n_shift.bit_len = len(kn2.cw_list)

    res_p = DPF.eval(res2_p_shift, kp2, party_p.party_id)
    res_n = DPF.eval(res2_n_shift, kn2, party_n.party_id)

    return res_p, res_n


def equal_match_key_gen_one_msb(num_of_keys):
    r2 = RingTensor.random([num_of_keys, BIT_LEN], down_bound=0, upper_bound=BIT_LEN)
    r2.bit_len = int(math.log2(BIT_LEN))
    k20, k21 = DPF.gen(num_of_keys * BIT_LEN, r2, RingTensor.convert_to_ring(1))
    from NssMPC import ArithmeticSecretSharing
    r20, r21 = ArithmeticSecretSharing.share(r2, 2)

    r3 = RingTensor.random([num_of_keys], down_bound=0, upper_bound=BIT_LEN)
    r3.bit_len = int(math.log2(BIT_LEN))
    k30, k31 = DPF.gen(num_of_keys, r3, RingTensor.convert_to_ring(1))
    r30, r31 = ArithmeticSecretSharing.share(r3, 2)
    return (k20, r20, k30, r30), (k21, r21, k31, r31)


def equal_match_one_msb(x, y, key):
    k2, r2, k3, r3 = key
    r2.party = r3.party = x.party

    delta = x ^ y

    delta = _to_bin(delta)

    delta = b2a(delta, x.party)

    res1 = delta.sum(-1)

    res1_shift_shared = r2 + res1
    res1_shift = res1_shift_shared.restore()

    res1_shift.bit_len = len(k2.cw_list)
    res2 = DPF.eval(res1_shift, k2, x.party.party_id)
    res2 = res2.sum(-1)

    from NssMPC import ArithmeticSecretSharing
    res2_shift_shared = r3 + ArithmeticSecretSharing(res2, x.party)
    res2_shift_shared -= RingTensor.convert_to_ring(1)
    res2_shift = res2_shift_shared.restore()
    res2_shift.bit_len = len(k3.cw_list)
    res = DPF.eval(res2_shift, k3, x.party.party_id)
    return ArithmeticSecretSharing(res, x.party)


def _split_num(x, bit_len):
    x = x.reshape(x.shape[0], -1, 1)
    small_x = x % (2 ** bit_len)
    x_ = x >> bit_len
    num = x.bit_len // bit_len
    for _ in range(num - 1):
        small_x = RingTensor.cat([small_x, x_ % (2 ** bit_len)], dim=-1)
        x_ = x_ >> bit_len
    small_x.bit_len = bit_len
    return small_x


def _to_bin(x: RingTensor):
    mask = 1
    shifted = x.unsqueeze(-1) >> RingTensor.arange(BIT_LEN - 1, -1, -1)
    # 使用按位与操作获取每个位置的二进制位
    binary_matrix = (shifted & mask)
    return binary_matrix
