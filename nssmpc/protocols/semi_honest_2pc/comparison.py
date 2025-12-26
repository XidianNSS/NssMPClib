#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import math
import os
import random
import re
from typing import Tuple, List, Iterator

from nssmpc.config import BIT_LEN, DEVICE
from nssmpc.infra.mpc.aux_parameter.parameter import Parameter, param_path
from nssmpc.infra.mpc.party import PartyCtx
from nssmpc.infra.mpc.party.party import Party
from nssmpc.infra.tensor import RingTensor
from nssmpc.primitives.homomorphic_encryption import Paillier
from nssmpc.primitives.secret_sharing.arithmetic import AdditiveSecretSharing
from nssmpc.primitives.secret_sharing.function import DICF, DPF, DICFKey, GrottoDICF, GrottoDICFKey, SigmaDICF, \
    SigmaDICFKey
from nssmpc.protocols.semi_honest_2pc.b2a import sh2pc_b2a


def sh2pc_eq(x: AdditiveSecretSharing, y: AdditiveSecretSharing, party: Party = None) -> AdditiveSecretSharing:
    """
    Secure comparison of two secret values for equality using Distributed Point Functions (DPF).

    The result determines if the difference :math:`d = x - y` is equal to 0.
    A random mask ``r_in`` is added to blind the difference to ensure input privacy.

    * If :math:`x = y`, returns a secret-shared value of 1.
    * Otherwise, returns a secret-shared value of 0.

    Args:
        x: The first secret-shared value to be compared.
        y: The second secret-shared value to be compared.
        party: The party instance managing the communication. Defaults to None.

    Returns:
        AdditiveSecretSharing: The secret-shared result of the equality check.

    Examples:
        >>> res = sh2pc_eq(x, y)
    """
    if party is None:
        party = PartyCtx.get()
    z = x - y
    shape = z.shape
    key = party.get_param(GrottoDICFKey, x.numel())
    z_shift = x.__class__(key.r_in) + z.flatten()
    z_shift = z_shift.recon()
    z_shift.dtype = 'int'
    ge_res = DPF.eval(z_shift.view(-1, 1), key.dpf_key, party.party_id).view(shape)
    ge_res.dtype = x.dtype
    return x.__class__(ge_res * x.scale)


def sh2pc_ge_msb(x: AdditiveSecretSharing, y: AdditiveSecretSharing, party: Party = None) -> AdditiveSecretSharing:
    """
    Secure greater-than-or-equal-to comparison (x >= y) using MSB extraction.

    This function computes the comparison by extracting the Most Significant Bit (MSB) of the difference :math:`x - y`.

    Args:
        x: The secret-shared value x.
        y: The secret-shared value y.
        party: The party instance managing the communication. Defaults to None.

    Returns:
        AdditiveSecretSharing: A secret sharing of the result (1 if :math:`x \\ge y`, else 0).

    Examples:
        >>> res = sh2pc_ge_msb(x, y)
    """
    if party is None:
        party = PartyCtx.get()
    z = x - y
    shape = z.shape
    msb = get_msb(z.item, party)
    ge_res = msb
    if party.party_id == 0:
        ge_res = msb ^ 1
    ge_res = sh2pc_b2a(ge_res, party)
    ge_res = ge_res.reshape(shape)
    ge_res.dtype = x.dtype
    ge_res = ge_res * x.scale
    return ge_res


def sh2pc_ge_dicf(x: AdditiveSecretSharing, y: AdditiveSecretSharing, party: Party = None) -> AdditiveSecretSharing:
    """
    Secure greater-than-or-equal-to comparison using DICF (Distributed Interval Comparison Function).

    This method utilizes Function Secret Sharing (FSS) or similar techniques to evaluate the comparison function efficiently.

    Args:
        x: The secret-shared value x.
        y: The secret-shared value y.
        party: The party instance managing the communication. Defaults to None.

    Returns:
        AdditiveSecretSharing: A secret sharing of the result (1 if :math:`x \\ge y`, else 0).

    Examples:
        >>> res = sh2pc_ge_dicf(x, y)
    """
    if party is None:
        party = PartyCtx.get()
    z = x - y
    shape = z.shape
    key = party.get_param(DICFKey, x.numel())
    z_shift = x.__class__(key.r_in) + z.flatten()
    z_shift = z_shift.recon()
    z_shift.dtype = 'int'
    ge_res = DICF.eval(z_shift, key, party.party_id).view(shape) * x.scale
    ge_res.dtype = x.dtype
    return x.__class__(ge_res)


def sh2pc_ge_ppq(x: AdditiveSecretSharing, y: AdditiveSecretSharing, party: Party = None) -> AdditiveSecretSharing:
    """
    Secure greater-than-or-equal-to comparison using PPQ (Prefix Parity Query).

    This method typically involves bit-decomposition or prefix-sum protocols to determine the relationship between x and y.

    Args:
        x: The secret-shared value x.
        y: The secret-shared value y.
        party: The party instance managing the communication. Defaults to None.

    Returns:
        AdditiveSecretSharing: A secret sharing of the result (1 if :math:`x \\ge y`, else 0).

    Examples:
        >>> res = sh2pc_ge_ppq(x, y)
    """
    if party is None:
        party = PartyCtx.get()
    z = x - y
    shape = z.shape
    key = party.get_param(GrottoDICFKey, x.numel())
    z_shift = x.__class__(key.r_in) - z.flatten()
    z_shift = z_shift.recon()
    z_shift.dtype = 'int'
    ge_res = GrottoDICF.eval(z_shift, key, party.party_id).view(shape)
    ge_res = sh2pc_b2a(ge_res, party)
    ge_res.dtype = x.dtype
    return ge_res * x.scale


def sh2pc_ge_sigma(x: AdditiveSecretSharing, y: AdditiveSecretSharing, party: Party = None) -> AdditiveSecretSharing:
    r"""
    Secure greater-than-or-equal-to comparison (x >= y) using the Sigma protocol.

    This function implements the comparison logic :math:`x \ge y` using the Sigma protocol (often involving zero-knowledge proof concepts or specific secret sharing extensions).

    Args:
        x: The secret-shared value x.
        y: The secret-shared value y.
        party: The party instance managing the communication. Defaults to None.

    Returns:
        AdditiveSecretSharing: A secret sharing of the result (1 if :math:`x \ge y`, else 0).

    Examples:
        >>> res = sh2pc_ge_sigma(x, y)
    """
    if party is None:
        party = PartyCtx.get()
    z = x - y
    shape = z.shape
    key = party.get_param(SigmaDICFKey, x.numel())
    z_shift = x.__class__(key.r_in) + z.flatten()
    z_shift = z_shift.recon()
    z_shift.dtype = 'int'
    ge_res = SigmaDICF.eval(z_shift, key, party.party_id).view(shape)
    ge_res = sh2pc_b2a(ge_res, party)
    ge_res.dtype = x.dtype
    return ge_res * x.scale


def get_msb(x: RingTensor, party: Party) -> RingTensor:
    """
    Extract the Most Significant Bit (MSB) from the input RingTensor ``x``.

    This is a low-level operation to retrieve the sign bit or MSB from the tensor representation.

    Args:
        x: The input RingTensor from which to extract the MSB.
        party: The party instance managing the communication.

    Returns:
        RingTensor: A RingTensor containing the MSB values.

    Examples:
        >>> msb = get_msb(x, party)
    """
    x = x.clone()
    shape = x.shape
    size = x.numel()
    x = _int2bit(x, size)
    carry_bit = _get_carry_bit(x, size, party)
    msb = carry_bit ^ x[:, -1]
    return msb.reshape(shape)


def _int2bit(x: RingTensor, size: int) -> RingTensor:
    """
    Convert an integer-based secret-shared RinTensor into its binary representation.

    This function takes an ArithmeticSharedRingTensor `x`, which contains secret-shared integers,
    and converts each element into its binary form (represented as a tensor with `BIT_LEN` bits).
    The result is a two-dimensional RingTensor where each row corresponds to the binary bits of each element in `x`.

    Args:
        x: An ASS containing secret-shared integers.
        size: The number of elements in the input `x`.

    Returns:
        RingTensor: A two-dimensional RingTensor where each row contains the binary representation (in `BIT_LEN` bits) of
            the corresponding integer in the original tensor `x`.

    Examples:
        >>> bits = _int2bit(x, size)
    """

    values = x.reshape(1, size)

    arr = RingTensor.zeros(size=(size, BIT_LEN))
    for i in range(0, BIT_LEN):
        arr[:, i] = ((values >> i) & 0x01).reshape(1, size)

    return arr


def _get_carry_bit(x: RingTensor, size: int, party: Party) -> RingTensor:
    """
    Get the carry bit of input `x`.

    Using the idea from the `Sonic paper <https://ieeexplore.ieee.org/document/9674792>`_,
    parallelization is used to speed up the acquisition of the highest carry bit.
    Introduce two parameters P and G，where P_i=a_i+b_i，G_i=a_i·b_i
    For the party 0, a is the binary representation of x and b is 0
    For the party 1, a is 0 and b is the binary representation of x

    Args:
        x: The input to get to carry bit.
        size: The number of integers to get the carry bit.
        party: The party instance.

    Returns:
        RingTensor: The secret-shared value of the carry bit.

    Examples:
        >>> carry = _get_carry_bit(x, size, party)
    """

    # layer = 0
    b = RingTensor.zeros(size=x.shape, device=DEVICE)
    p_layer0 = x ^ b
    g_layer0 = _get_g(x, size, party)

    # layer = 1
    p_temp, g_temp = _get_p_and_g(p_layer0, g_layer0, party, BIT_LEN - 1, BIT_LEN // 2 - 1, True)
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
        p_layer, g_layer = _get_p_and_g(p_layer, g_layer, party, BIT_LEN // (2 ** i),
                                        BIT_LEN // (2 ** (i + 1)),
                                        False)

    carry_bit = _get_g_last_layer(p_layer, g_layer, party, 2, 1)
    carry_bit = carry_bit.reshape(carry_bit.size()[0])

    return carry_bit


def _get_g(x: RingTensor, size: int, party: Party) -> RingTensor:
    """
    Get the parameter G of the first level.

    Args:
        x: The input to get the parameter G.
        size: The number of integers to get the parameter G.
        party: The party instance.

    Returns:
        RingTensor: The parameter G.

    Examples:
        >>> g = _get_g(x, size, party)
    """

    a, b, c = party.get_param(BooleanTriples, BIT_LEN)
    a = a.to(DEVICE)
    b = b.to(DEVICE)
    c = c.to(DEVICE)

    x_prime = RingTensor.zeros(size=(size, BIT_LEN), device=DEVICE)

    if party.party_id == 0:
        e = x ^ a
        f = x_prime ^ b
    else:
        e = x_prime ^ a
        f = x ^ b

    party.send(RingTensor.cat([e, f], dim=0))
    get_array = party.recv()

    length = int(get_array.shape[0] / 2)

    e_i = get_array[:length]
    f_i = get_array[length:]

    common_e = e ^ e_i
    common_f = f ^ f_i

    return (RingTensor(party.party_id, dtype=x.dtype).to(DEVICE) & common_f & common_e) \
        ^ (common_e & b) ^ (common_f & a) ^ c


def _get_p_and_g(p: RingTensor, g: RingTensor, party, in_num: int, out_num: int, is_layer1: bool) -> Tuple[
    RingTensor, RingTensor]:
    """
    Compute the P and G of next level according to the current level.

    Args:
        p: The P parameter of the current level.
        g: The G parameter of the current level.
        party (Party): The party of the computation.
        in_num: The number of input.
        out_num: The number of output.
        is_layer1: The sign to judge whether the current level is the first level.

    Returns:
        Tuple[RingTensor, RingTensor]: The P and G of next level.

    Examples:
        >>> p_next, g_next = _get_p_and_g(p, g, party, in_num, out_num, is_layer1)
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

    party.send(RingTensor.cat([e_p1, f_p2_p, e_g1, f_p2_g], dim=1))
    get_array = party.recv()

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


def _get_g_last_layer(p: RingTensor, g: RingTensor, party: Party, in_num: int, out_num: int) -> RingTensor:
    """
    Get the parameter G of the last level

    Args:
        p: The P parameter of the second to last level.
        g: The G parameter of the second to last level.
        party: The party of the computation.
        in_num: The number of input.
        out_num: The number of output.

    Returns:
        RingTensor: The P and G of next level.

    Examples:
        >>> g_last = _get_g_last_layer(p, g, party, in_num, out_num)
    """
    p_in2 = p[:, 1: in_num: 2]
    g_in1 = g[:, 0: in_num: 2]
    g_in2 = g[:, 1: in_num: 2]

    a_g1, b_p2_g, c_g1_p2 = party.get_param(BooleanTriples, out_num)

    e_g1 = a_g1 ^ g_in1
    f_p2_g = b_p2_g ^ p_in2

    party.send(RingTensor.cat([e_g1, f_p2_g], dim=1))
    get_array = party.recv()

    out_num = int(get_array.shape[1] / 2)

    e_i = get_array[:, :out_num]
    f_i = get_array[:, out_num:]

    common_e = e_g1 ^ e_i
    common_f = f_p2_g ^ f_i

    g_out = (RingTensor(party.party_id, dtype=g.dtype).to(DEVICE) & common_f & common_e) ^ (common_e & b_p2_g) ^ (
            common_f & a_g1) ^ c_g1_p2
    g_out = g_out ^ g_in2

    return g_out


class BooleanTriples(Parameter):
    """
    This is a parameter class for handling Boolean triples, which allows the generation and saving of Boolean triples.
    """

    def __init__(self, a: RingTensor = None, b: RingTensor = None, c: RingTensor = None):
        """
        Initializes the a, b, and c of the multiplicative triple and sets the size of the elements in the triple to 0.
        """
        self.a = a
        self.b = b
        self.c = c
        self.size = 0

    def __iter__(self) -> Iterator[RingTensor]:
        """
        Iterator support.

        Allows the BooleanTriples instance to be iterated over, returning each element of the triple.

        Returns:
            iterator: Returns an iterator for boolean triples.
        """
        return iter((self.a, self.b, self.c))

    @staticmethod
    def gen(num_of_triples: int, num_of_party: int = 2, type_of_generation: str = 'TTP',
            party=None):
        """
        Generate a specified number of Boolean triples according to different generation methods.

        Call the corresponding function based on the value of ``type_of_generation`` to generate Boolean triples:
            * :func:`gen_msb_triples_by_homomorphic_encryption` : Generate triples based on homomorphic encryption.
            * :func:`gen_msb_triples_by_ttp` : Generates triples based on trusted third parties.

        Args:
            num_of_triples: number of triples
            num_of_party: number of parties
            type_of_generation: generation type. (TTP: generated by trusted third party, HE: generated by homomorphic encryption)
            party (Party, optional): party if HE

        Returns:
            List['BooleanTriples'] | 'BooleanTriples': The generated triples.

        Examples:
            >>> triples = BooleanTriples.gen(100)
        """
        if type_of_generation == 'HE':
            return gen_msb_triples_by_homomorphic_encryption(num_of_triples, party)
        elif type_of_generation == 'TTP':
            return gen_msb_triples_by_ttp(num_of_triples, num_of_party)


def gen_msb_triples_by_ttp(bit_len: int, num_of_party: int = 2) -> List[BooleanTriples]:
    """
    Generate the multiplication Beaver triple by trusted third party.

    First, two binary tensors, ``a`` and ``b``, are randomly generated with values between 0 and 1. Then ``c = a & b`` is
    calculated, that is, the bitwise and operation is performed. ``a``, ``b``, and ``c`` are shared to each participant using
    Boolean Secret Sharing, and each participant's shared value is converted to a tensor on the *CPU* device and stored
    in the triples list.

    Args:
        bit_len: the length of the binary string
        num_of_party: the number of parties

    Returns:
        list: the list of msb triples generated by TTP.

    Examples:
        >>> triples = gen_msb_triples_by_ttp(32, 2)
    """
    a = RingTensor.random([bit_len], down_bound=0, upper_bound=2, device=DEVICE)
    b = RingTensor.random([bit_len], down_bound=0, upper_bound=2, device=DEVICE)
    c = a & b
    from nssmpc.primitives.secret_sharing.boolean import BooleanSecretSharing
    a_list = BooleanSecretSharing.share(a, num_of_party)
    b_list = BooleanSecretSharing.share(b, num_of_party)
    c_list = BooleanSecretSharing.share(c, num_of_party)

    triples = []
    for i in range(num_of_party):
        triples.append(BooleanTriples(a_list[i].to(DEVICE), b_list[i].to(DEVICE), c_list[i].to(DEVICE)))
        triples[i].size = bit_len

    return triples


def gen_msb_triples_by_homomorphic_encryption(bit_len: int, party: Party) -> BooleanTriples:
    """
    Generate the multiplication Beaver triple by homomorphic encryption.

    First, two binary lists a and b are randomly generated, along with an empty list **c**.
        If the current participant`s party_id is **0**:
            First, the key of Paillier homomorphic encryption is generated to encrypt a and b, and then the encrypted
            value is sent to other participants, and the data from other participants is received, and the value of c is
            decrypted.

        If the current participant`s party_id is **1**:
            The random value ``r`` is first generated to compute the
            partial value of ``c``. After receiving the message from Party 0, ``r`` is encrypted using homomorphic encryption,
            and a new encrypted value ``d`` is calculated based on the received data, and then sent back.

    Args:
        bit_len: the length of the binary string
        party: the party to generate triples

    Returns:
        BooleanTriples: A BooleanTriples that includes a, b, and c.

    Examples:
        >>> triple = gen_msb_triples_by_homomorphic_encryption(32, party)
    """

    a = [random.randint(0, 2) for _ in range(bit_len)]
    b = [random.randint(0, 2) for _ in range(bit_len)]
    c = []

    if party.party_id == 0:
        paillier = Paillier()
        paillier.gen_keys()

        encrypted_a = paillier.encrypt(a)
        encrypted_b = paillier.encrypt(b)
        party.send([encrypted_a, encrypted_b, paillier.public_key])
        d = party.recv()
        decrypted_d = paillier.decrypt(d)
        c = [decrypted_d[i] + a[i] * b[i] for i in range(bit_len)]

    elif party.party_id == 1:
        r = [random.randint(0, 2) for _ in range(bit_len)]
        c = [a[i] * b[i] - r[i] for i in range(bit_len)]

        messages = party.recv()

        encrypted_r = Paillier.encrypt_with_key(r, messages[2])
        d = [messages[0][i] ** b[i] * messages[1][i] ** a[i] * encrypted_r[i] for i in range(bit_len)]
        party.send(d)

    msb_triples = BooleanTriples(RingTensor(a).to('cpu'), RingTensor(b).to('cpu'),
                                 RingTensor(c).to('cpu'))
    return msb_triples
