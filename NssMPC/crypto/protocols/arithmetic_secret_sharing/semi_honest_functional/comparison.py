#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import math

from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.aux_parameter import BooleanTriples, GrottoDICFKey, SigmaDICFKey, DICFKey

from NssMPC import RingTensor
from NssMPC.config import GE_TYPE, BIT_LEN, DEVICE
from NssMPC.crypto.primitives.function_secret_sharing import *
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional.b2a import b2a


def secure_eq(x, y):
    """
    Secure comparison of two secret values for equality using DPF.

    The final result is obtained by comparing whether the difference between the two inputs `x` and `y` is equal to 0.
    A random value `r_in` is added to blind the difference between the two inputs (x and y) to ensure the security of the inputs.
    If they are equal, it returns a secret-shared value of 1; otherwise, it returns a secret-shared value of 0.

    :param x: The secret-shared value to be compared
    :type x: ArithmeticSecretSharing
    :param y: The other secret-shared value to be compared
    :type y: ArithmeticSecretSharing
    :return: The secret-shared result of the comparison
    :rtype: ArithmeticSecretSharing
    """
    z = x - y
    shape = z.shape
    key = PartyRuntime.party.get_param(GrottoDICFKey, x.numel())
    z_shift = x.__class__(key.r_in) + z.flatten()
    z_shift = z_shift.restore()
    z_shift.dtype = 'int'
    ge_res = DPF.eval(z_shift.view(-1, 1), key.dpf_key, PartyRuntime.party.party_id).view(shape)
    ge_res.dtype = x.dtype
    return x.__class__(ge_res * x.scale)


def secure_ge(x, y):
    """
    Secure greater-than-or-equal-to comparison.

    The greater-than-or-equal-to comparison is implemented in four different ways. If `x` is greater than or equal to `y`,
    it returns a secret-shared value of 1; otherwise, it returns a secret-shared value of 0.
    The comparison method is chosen by changing the value of `GE_TYPE`.

    :param x: The secret-shared value to be compared
    :type x: ArithmeticSecretSharing
    :param y: The other secret-shared value to be compared
    :type y: ArithmeticSecretSharing
    :return: The secret-shared result of the comparison
    :rtype: ArithmeticSecretSharing
    """
    ge_methods = {'MSB': msb_ge, 'DICF': dicf_ge, 'PPQ': ppq_ge, 'SIGMA': sigma_ge}
    return ge_methods[GE_TYPE](x, y)


def msb_ge(x, y):
    """
    Secure greater-than-or-equal-to comparison using MSB (most significant bit).

    The comparison is implemented using the most significant bit (MSB) method.
    If `x` is greater than or equal to `y`, it returns a secret-shared value of 1; otherwise, it returns a secret-shared value of 0.

    :param x: The secret-shared value to be compared
    :type x: ArithmeticSecretSharing
    :param y: The other secret-shared value to be compared
    :type y: ArithmeticSecretSharing
    :return: The secret-shared result of the comparison
    :rtype: ArithmeticSecretSharing
    """
    z = x - y
    shape = z.shape
    msb = get_msb(z)
    ge_res = msb
    if PartyRuntime.party.party_id == 0:
        ge_res = msb ^ 1
    ge_res = b2a(ge_res, PartyRuntime.party)
    ge_res = ge_res.reshape(shape)
    ge_res.dtype = x.dtype
    ge_res = ge_res * x.scale
    return ge_res


def dicf_ge(x, y):
    """
    Secure greater-than-or-equal-to comparison using DICF

    The comparison is implemented using the DICF method. If `x` is greater than or equal to `y`,
    it returns a secret-shared value of 1; otherwise, it returns a secret-shared value of 0.

    :param x: The secret-shared value to be compared
    :type x: ArithmeticSecretSharing
    :param y: The other secret-shared value to be compared
    :type y: ArithmeticSecretSharing
    :return: The secret-shared result of the comparison
    :rtype: ArithmeticSecretSharing
    """
    z = x - y
    shape = z.shape
    key = PartyRuntime.party.get_param(DICFKey, x.numel())
    z_shift = x.__class__(key.r_in, PartyRuntime.party) + z.flatten()
    z_shift = z_shift.restore()
    z_shift.dtype = 'int'
    ge_res = DICF.eval(z_shift, key, PartyRuntime.party.party_id).view(shape) * x.scale
    ge_res.dtype = x.dtype
    return x.__class__(ge_res, PartyRuntime.party)


def ppq_ge(x, y):
    """
    Secure greater-than-or-equal-to comparison using PPQ (prefix parity query)

    The comparison is implemented using the PPQ method. If `x` is greater than or equal to `y`,
    it returns a secret-shared value of 1; otherwise, it returns a secret-shared value of 0.

    :param x: The secret-shared value to be compared
    :type x: ArithmeticSecretSharing
    :param y: The other secret-shared value to be compared
    :type y: ArithmeticSecretSharing
    :return: The secret-shared result of the comparison
    :rtype: ArithmeticSecretSharing
    """
    z = x - y
    shape = z.shape
    key = PartyRuntime.party.get_param(GrottoDICFKey, x.numel())
    z_shift = x.__class__(key.r_in, PartyRuntime.party) - z.flatten()
    z_shift = z_shift.restore()
    z_shift.dtype = 'int'
    ge_res = GrottoDICF.eval(z_shift, key, PartyRuntime.party.party_id).view(shape)
    ge_res = b2a(ge_res, PartyRuntime.party)
    ge_res.dtype = x.dtype
    return ge_res * x.scale


def sigma_ge(x, y):
    """
    Secure greater-than-or-equal-to comparison using Sigma

    The comparison is implemented using the Sigma method. If `x` is greater than or equal to `y`,
    it returns a secret-shared value of 1; otherwise, it returns a secret-shared value of 0.

    :param x: The secret-shared value to be compared
    :type x: ArithmeticSecretSharing
    :param y: The other secret-shared value to be compared
    :type y: ArithmeticSecretSharing
    :return: The secret-shared result of the comparison
    :rtype: ArithmeticSecretSharing
    """
    z = x - y
    shape = z.shape
    key = PartyRuntime.party.get_param(SigmaDICFKey, x.numel())
    z_shift = x.__class__(key.r_in) + z.flatten()
    z_shift = z_shift.restore()
    z_shift.dtype = 'int'
    ge_res = SigmaDICF.eval(z_shift, key, PartyRuntime.party.party_id).view(shape)
    ge_res = b2a(ge_res, PartyRuntime.party)
    ge_res.dtype = x.dtype
    return ge_res * x.scale


def get_msb(x) -> RingTensor:
    """
    Extract the `msb(most significant bit)` value from input `x`.

    :param x: The input RingTensor to be extracted.
    :type x: RingTensor
    :return: The secret-shared value of the `msb`.
    :rtype: RingTensor
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
    Convert an integer-based secret-shared RinTensor into its binary representation.

    This function takes an ArithmeticSharedRingTensor `x`, which contains secret-shared integers,
    and converts each element into its binary form (represented as a tensor with `BIT_LEN` bits).
    The result is a two-dimensional RingTensor where each row corresponds to the binary bits of each element in `x`.

    :param x: An ASS containing secret-shared integers.
    :type x: ASS
    :param size: The number of elements in the input `x`.
    :type size: int

    :return: A two-dimensional RingTensor where each row contains the binary representation (in `BIT_LEN` bits) of
            the corresponding integer in the original tensor `x`.
    :rtype: RingTensor
    """

    values = x.reshape(1, size).item

    arr = RingTensor.zeros(size=(size, BIT_LEN))
    for i in range(0, BIT_LEN):
        arr[:, i] = ((values >> i) & 0x01).reshape(1, size)

    return x.__class__(arr, PartyRuntime.party)


def _get_carry_bit(x, size: int) -> RingTensor:
    """
    Get the carry bit of input `x`.

    Using the idea from the `Sonic paper <https://ieeexplore.ieee.org/document/9674792>`_,
    parallelization is used to speed up the acquisition of the highest carry bit.
    Introduce two parameters P and G，where P_i=a_i+b_i，G_i=a_i·b_i
    For the party 0, a is the binary representation of x and b is 0
    For the party 1, a is 0 and b is the binary representation of x

    :param x: The input to get to carry bit.
    :type x: RingTensor
    :param size: The number of integers to get the carry bit.
    :type size: int
    :return: The secret-shared value of the carry bit.
    :rtype: RingTensor
    """

    # layer = 0
    b = RingTensor.zeros(size=x.shape, device=DEVICE)
    p_layer0 = x.item ^ b
    g_layer0 = _get_g(x, size)

    # layer = 1
    p_temp, g_temp = _get_p_and_g(p_layer0, g_layer0, PartyRuntime.party, BIT_LEN - 1, BIT_LEN // 2 - 1, True)
    p_layer1 = RingTensor.zeros(size=(size, BIT_LEN // 2), device=DEVICE)
    g_layer1 = RingTensor.zeros(size=(size, BIT_LEN // 2), device=DEVICE)

    p_layer1[:, 1:] = p_temp
    g_layer1[:, 1:] = g_temp
    p_layer1[:, 0] = p_layer0[:, 0]
    g_layer1[:, 0] = g_layer0[:, 0]

    p_layer = p_layer1
    g_layer = g_layer1

    layer_total = int(math.log2(BIT_LEN))

    for i in range(1, layer_total - 1):
        p_layer, g_layer = _get_p_and_g(p_layer, g_layer, PartyRuntime.party, BIT_LEN // (2 ** i),
                                        BIT_LEN // (2 ** (i + 1)),
                                        False)

    carry_bit = _get_g_last_layer(p_layer, g_layer, PartyRuntime.party, 2, 1)
    carry_bit = carry_bit.reshape(carry_bit.size()[0])

    return carry_bit


def _get_g(x, size: int) -> RingTensor:
    """
    Get the parameter G of the first level.

    :param x: The input to get the parameter G.
    :type x: RingTensor
    :param size: The number of integers to get the parameter G.
    :type size: int
    :return: The parameter G.
    :rtype: RingTensor
    """

    a, b, c = PartyRuntime.party.get_param(BooleanTriples, BIT_LEN)
    a = a.to(DEVICE)
    b = b.to(DEVICE)
    c = c.to(DEVICE)

    x_prime = RingTensor.zeros(size=(size, BIT_LEN), device=DEVICE)

    if PartyRuntime.party.party_id == 0:
        e = x.item ^ a
        f = x_prime ^ b
    else:
        e = x_prime ^ a
        f = x.item ^ b

    PartyRuntime.party.send(RingTensor.cat((e, f), dim=0))
    get_array = PartyRuntime.party.receive()

    length = int(get_array.shape[0] / 2)

    e_i = get_array[:length]
    f_i = get_array[length:]

    common_e = e ^ e_i
    common_f = f ^ f_i

    return (RingTensor(PartyRuntime.party.party_id, dtype=x.dtype).to(DEVICE) & common_f & common_e) \
        ^ (common_e & b) ^ (common_f & a) ^ c


def _get_p_and_g(p: RingTensor, g: RingTensor, party, in_num, out_num, is_layer1):
    """
    Compute the P and G of next level according to the current level.

    :param p: The P parameter of the current level.
    :type p: RingTensor
    :param g: The G parameter of the current level.
    :type g: RingTensor
    :param party: The party of the computation.
    :type party: Party
    :param in_num: The number of input.
    :type in_num: int
    :param out_num: The number of output.
    :type out_num: int
    :param is_layer1: The sign to judge whether the current level is the first level.
    :type is_layer1: bool
    :return: The P and G of next level.
    :rtype: RingTensor
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
    """
    Get the parameter G of the last level

    :param p: The P parameter of the second to last level.
    :type p: RingTensor
    :param g: The G parameter of the second to last level.
    :type g: RingTensor
    :param party: The party of the computation.
    :type party: Party
    :param in_num: The number of input.
    :type in_num: int
    :param out_num: The number of output.
    :type out_num: int
    :return: The P and G of next level.
    :rtype: RingTensor
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
