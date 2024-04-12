"""
Implement msb based on sonic
Refer to Liu e.t.c Securely Outsourcing Neural Network Inference to the Cloud with Lightweight Techniques
https://ieeexplore.ieee.org/abstract/document/9674792
"""

from common.tensor import *
from config.base_configs import *
from crypto.primitives.beaver.msb_triples import MSBTriples


def get_msb(x) -> RingTensor:
    """
    Extract the msb(most significant bit) value from x

    Returns:
        msb(most significant bit)
    """
    x = x.clone()
    shape = x.shape
    size = x.numel()
    x = int2bit(x, size)
    carry_bit = get_carry_bit(x, size)
    msb = carry_bit ^ x.ring_tensor[:, -1]
    return msb.reshape(shape)


def int2bit(x, size: int):
    """
    Get a one-dimensional ArithmeticSharedRingTensor in binary form
    equal to the tensor size of x and the number of elements of x

    Args:
        x: input
        size: the size of the x

    Returns:
        a one-dimensional binary representation of x
    """

    values = x.reshape(1, size).ring_tensor

    arr = RingFunc.zeros(size=(size, BIT_LEN))
    for i in range(0, BIT_LEN):
        arr[:, i] = ((values >> i) & 0x01).reshape(1, size)

    return x.__class__(arr, x.party)


def get_carry_bit(x, size: int) -> RingTensor:
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
    b = RingFunc.zeros(size=x.shape, device=DEVICE)
    p_layer0 = x.ring_tensor ^ b
    g_layer0 = get_g(x, size)

    # layer = 1
    p_temp, g_temp = get_p_and_g(p_layer0, g_layer0, x.party, BIT_LEN - 1, BIT_LEN // 2 - 1, True)
    p_layer1 = RingFunc.zeros(size=(size, BIT_LEN // 2), device=DEVICE)
    g_layer1 = RingFunc.zeros(size=(size, BIT_LEN // 2), device=DEVICE)

    p_layer1[:, 1:] = p_temp
    g_layer1[:, 1:] = g_temp
    p_layer1[:, 0] = p_layer0[:, 0]
    g_layer1[:, 0] = g_layer0[:, 0]

    p_layer = p_layer1
    g_layer = g_layer1

    layer_total = int(math.log2(BIT_LEN))

    for i in range(2, layer_total - 1):
        p_layer, g_layer = get_p_and_g(p_layer, g_layer, x.party, BIT_LEN // (2 ** i), BIT_LEN // (2 ** (i + 1)), False)

    carry_bit = get_g_last_layer(p_layer, g_layer, x.party, 2, 1)
    carry_bit = carry_bit.reshape(carry_bit.size()[0])

    return carry_bit


def get_g(x, size: int) -> RingTensor:
    """
    Get the parameter G of the first level

    Args:
        x: input
        size: size of the x

    Returns:
        parameter G of the first level
    """

    a, b, c = x.party.get_param(MSBTriples, BIT_LEN)
    a = a.to(DEVICE)
    b = b.to(DEVICE)
    c = c.to(DEVICE)

    x_prime = RingFunc.zeros(size=(size, BIT_LEN), device=DEVICE)

    if x.party.party_id == 0:
        e = x.ring_tensor ^ a
        f = x_prime ^ b
    else:
        e = x_prime ^ a
        f = x.ring_tensor ^ b

    x.party.send(RingTensor.cat((e, f), dim=0))
    get_array = x.party.receive()

    length = int(get_array.shape[0] / 2)

    e_i = get_array[:length]
    f_i = get_array[length:]

    common_e = e ^ e_i
    common_f = f ^ f_i

    return (RingTensor(x.party.party_id, dtype=x.dtype).to(DEVICE) & common_f & common_e) \
        ^ (common_e & b) ^ (common_f & a) ^ c


def get_p_and_g(p: RingTensor, g: RingTensor, party, in_num, out_num, is_layer1):
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

    a_p1, b_p2_p, c_p1_p2 = party.get_param(MSBTriples, out_num)
    a_g1, b_p2_g, c_g1_p2 = party.get_param(MSBTriples, out_num)

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


def get_g_last_layer(p: RingTensor, g: RingTensor, party, in_num, out_num) -> RingTensor:
    """Get the parameter G of the last level
    """
    p_in2 = p[:, 1: in_num: 2]
    g_in1 = g[:, 0: in_num: 2]
    g_in2 = g[:, 1: in_num: 2]

    a_g1, b_p2_g, c_g1_p2 = party.get_param(MSBTriples, out_num)

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
