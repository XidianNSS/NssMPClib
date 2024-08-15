import math

from NssMPC.crypto.aux_parameter import BooleanTriples, GrottoDICFKey, SigmaDICFKey, DICFKey

from NssMPC import RingTensor
from NssMPC.config import GE_TYPE, BIT_LEN, DEVICE
from NssMPC.crypto.primitives.function_secret_sharing import *
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional.b2a import b2a


def secure_eq(x, y):
    z = x - y
    shape = z.shape
    key = x.party.get_param(GrottoDICFKey, x.numel())
    z_shift = x.__class__(key.r_in, x.party) + z.flatten()
    z_shift = z_shift.restore()
    z_shift.dtype = 'int'
    ge_res = DPF.eval(z_shift.view(-1,1), key.dpf_key, x.party.party_id).view(shape)
    ge_res.dtype = x.dtype
    return x.__class__(ge_res * x.scale, x.party)


def secure_ge(x, y):
    """
    The comparison of two ArithmeticSharedRingTensor.

    There are four different ways you can choose to compare the two ArithmeticSharedRingTensor by changing the value of
    Ge_TYPE.
    """
    ge_methods = {'MSB': msb_ge, 'DICF': dicf_ge, 'PPQ': ppq_ge, 'SIGMA': sigma_ge}
    return ge_methods[GE_TYPE](x, y)


def msb_ge(x, y):
    z = x - y
    shape = z.shape
    msb = get_msb(z)
    ge_res = msb
    if x.party.party_id == 0:
        ge_res = msb ^ 1
    ge_res = b2a(ge_res, x.party)
    ge_res = ge_res.reshape(shape)
    ge_res.dtype = x.dtype
    ge_res = ge_res * x.scale
    return ge_res


def dicf_ge(x, y):
    z = x - y
    shape = z.shape
    key = x.party.get_param(DICFKey, x.numel())
    z_shift = x.__class__(key.r_in, x.party) + z.flatten()
    z_shift = z_shift.restore()
    z_shift.dtype = 'int'
    ge_res = DICF.eval(z_shift, key, x.party.party_id).view(shape) * x.scale
    ge_res.dtype = x.dtype
    return x.__class__(ge_res, x.party)


def ppq_ge(x, y):
    z = x - y
    shape = z.shape
    key = x.party.get_param(GrottoDICFKey, x.numel())
    z_shift = x.__class__(key.r_in, x.party) - z.flatten()
    z_shift = z_shift.restore()
    z_shift.dtype = 'int'
    ge_res = GrottoDICF.eval(z_shift, key, x.party.party_id).view(shape)
    ge_res = b2a(ge_res, x.party)
    ge_res.dtype = x.dtype
    return ge_res * x.scale


def sigma_ge(x, y):
    z = x - y
    shape = z.shape
    key = x.party.get_param(SigmaDICFKey, x.numel())
    z_shift = x.__class__(key.r_in, x.party) + z.flatten()
    z_shift = z_shift.restore()
    z_shift.dtype = 'int'
    ge_res = SigmaDICF.eval(z_shift, key, x.party.party_id).view(shape)
    ge_res = b2a(ge_res, x.party)
    ge_res.dtype = x.dtype
    return ge_res * x.scale


def get_msb(x) -> RingTensor:
    """
    Extract the msb(most significant bit) value from x

    Returns:
        msb(most significant bit)
    """
    x = x.clone()
    shape = x.shape
    size = x.numel()
    x = _int2bit(x, size)
    carry_bit = _get_carry_bit(x, size)
    msb = carry_bit ^ x.item[:, -1]
    return msb.reshape(shape)


def _int2bit(x, size: int):
    """
    Get a one-dimensional ArithmeticSharedRingTensor in binary form
    equal to the tensor size of x and the number of elements of x

    Args:
        x: input
        size: the size of the x

    Returns:
        a one-dimensional binary representation of x
    """

    values = x.reshape(1, size).item

    arr = RingTensor.zeros(size=(size, BIT_LEN))
    for i in range(0, BIT_LEN):
        arr[:, i] = ((values >> i) & 0x01).reshape(1, size)

    return x.__class__(arr, x.party)


def _get_carry_bit(x, size: int) -> RingTensor:
    """
    Get the carry bit
    Using the idea from the sonic paper, parallelization is used to speed up the acquisition of the highest carry bit
    Introduce two parameters P and G，where P_i=a_i+b_i，G_i=a_i·b_i
    For the party 0, a is the binary representation of x and b is 0
    For the party 1, a is 0 and b is the binary representation of x

    Args:
        x : input
        size: Binary bit length of the number

    Returns:
        the highest carry bit
    """

    # layer = 0
    b = RingTensor.zeros(size=x.shape, device=DEVICE)
    p_layer0 = x.item ^ b
    g_layer0 = _get_g(x, size)

    # layer = 1
    p_temp, g_temp = _get_p_and_g(p_layer0, g_layer0, x.party, BIT_LEN - 1, BIT_LEN // 2 - 1, True)
    p_layer1 = RingTensor.zeros(size=(size, BIT_LEN // 2), device=DEVICE)
    g_layer1 = RingTensor.zeros(size=(size, BIT_LEN // 2), device=DEVICE)

    p_layer1[:, 1:] = p_temp
    g_layer1[:, 1:] = g_temp
    p_layer1[:, 0] = p_layer0[:, 0]
    g_layer1[:, 0] = g_layer0[:, 0]

    p_layer = p_layer1
    g_layer = g_layer1

    layer_total = int(math.log2(BIT_LEN))

    for i in range(2, layer_total - 1):
        p_layer, g_layer = _get_p_and_g(p_layer, g_layer, x.party, BIT_LEN // (2 ** i), BIT_LEN // (2 ** (i + 1)),
                                        False)

    carry_bit = _get_g_last_layer(p_layer, g_layer, x.party, 2, 1)
    carry_bit = carry_bit.reshape(carry_bit.size()[0])

    return carry_bit


def _get_g(x, size: int) -> RingTensor:
    """
    Get the parameter G of the first level

    Args:
        x: input
        size: size of the x

    Returns:
        parameter G of the first level
    """

    a, b, c = x.party.get_param(BooleanTriples, BIT_LEN)
    a = a.to(DEVICE)
    b = b.to(DEVICE)
    c = c.to(DEVICE)

    x_prime = RingTensor.zeros(size=(size, BIT_LEN), device=DEVICE)

    if x.party.party_id == 0:
        e = x.item ^ a
        f = x_prime ^ b
    else:
        e = x_prime ^ a
        f = x.item ^ b

    x.party.send(RingTensor.cat((e, f), dim=0))
    get_array = x.party.receive()

    length = int(get_array.shape[0] / 2)

    e_i = get_array[:length]
    f_i = get_array[length:]

    common_e = e ^ e_i
    common_f = f ^ f_i

    return (RingTensor(x.party.party_id, dtype=x.dtype).to(DEVICE) & common_f & common_e) \
        ^ (common_e & b) ^ (common_f & a) ^ c


def _get_p_and_g(p: RingTensor, g: RingTensor, party, in_num, out_num, is_layer1):
    """
    Compute the P and G of next level according to the current level

    Args:
        p: the P parameter of the current level
        g: the G parameter of the current level
        party: the party of the computation
        in_num: the number of input
        out_num: the number of output
        is_layer1: the sign to judge whether the current level is the first level

    Returns:
        the P and G of next level
    """
    if is_layer1:
        start_bit = 1
    else:
        start_bit = 0

    p_in1 = p[:, start_bit: in_num: 2]
    p_in2 = p[:, start_bit + 1: in_num: 2]
    g_in1 = g[:, start_bit: in_num: 2]
    g_in2 = g[:, start_bit + 1: in_num: 2]

    a_p1, b_p2_p, c_p1_p2 = party.get_param(BooleanTriples, out_num)
    a_g1, b_p2_g, c_g1_p2 = party.get_param(BooleanTriples, out_num)

    e_p1 = a_p1 ^ p_in1
    f_p2_p = b_p2_p ^ p_in2
    e_g1 = a_g1 ^ g_in1
    f_p2_g = b_p2_g ^ p_in2

    party.send(RingTensor.cat((e_p1, f_p2_p, e_g1, f_p2_g), dim=1))
    get_array = party.receive()

    length = int(get_array.shape[1] / 4)

    e_i = get_array[:, :length]
    f_i = get_array[:, length: length * 2]

    common_e = e_p1 ^ e_i
    common_f = f_p2_p ^ f_i

    p_out = (RingTensor(party.party_id, dtype=p.dtype).to(DEVICE) & common_f & common_e) ^ (common_e & b_p2_p) ^ (
            common_f & a_p1) ^ c_p1_p2

    e_i = get_array[:, length * 2:length * 3]
    f_i = get_array[:, length * 3:]

    common_e = e_g1 ^ e_i
    common_f = f_p2_g ^ f_i

    g_out = (RingTensor(party.party_id, dtype=g.dtype).to(DEVICE) & common_f & common_e) ^ (common_e & b_p2_g) ^ (
            common_f & a_g1) ^ c_g1_p2
    g_out = g_out ^ g_in2

    return p_out, g_out


def _get_g_last_layer(p: RingTensor, g: RingTensor, party, in_num, out_num) -> RingTensor:
    """Get the parameter G of the last level
    """
    p_in2 = p[:, 1: in_num: 2]
    g_in1 = g[:, 0: in_num: 2]
    g_in2 = g[:, 1: in_num: 2]

    a_g1, b_p2_g, c_g1_p2 = party.get_param(BooleanTriples, out_num)

    e_g1 = a_g1 ^ g_in1
    f_p2_g = b_p2_g ^ p_in2

    party.send(RingTensor.cat((e_g1, f_p2_g), dim=1))
    get_array = party.receive()

    out_num = int(get_array.shape[1] / 2)

    e_i = get_array[:, :out_num]
    f_i = get_array[:, out_num:]

    common_e = e_g1 ^ e_i
    common_f = f_p2_g ^ f_i

    g_out = (RingTensor(party.party_id, dtype=g.dtype).to(DEVICE) & common_f & common_e) ^ (common_e & b_p2_g) ^ (
            common_f & a_g1) ^ c_g1_p2
    g_out = g_out ^ g_in2

    return g_out
