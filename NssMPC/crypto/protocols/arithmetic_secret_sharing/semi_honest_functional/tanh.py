#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.primitives.function_secret_sharing.dicf import SigmaDICF
from NssMPC import RingTensor
from NssMPC.config import SCALE_BIT, TANH_TABLE_BIT
from NssMPC.crypto.aux_parameter.look_up_table_keys.tanh_key import TanhKey
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional import b2a
from NssMPC.crypto.protocols.look_up_table import LookUp
from NssMPC.crypto.protocols.selection.selectlin import SelectLin


def secure_tanh(x):
    """
    Securely computes the hyperbolic tangent (tanh) of the input `x` using secret sharing and lookup tables.

    :param x: Input ASS(ArithmeticSecretSharing) to compute the tanh.
    :type x: ArithmeticSecretSharing
    :return: The securely computed tanh of the input `x`.
    :rtype: ArithmeticSecretSharing
    """
    table_scale_bit = TANH_TABLE_BIT
    shape = x.shape
    x = x.flatten()
    party = PartyRuntime.party
    tanh_key = party.get_param(TanhKey, x.numel())
    sigma_key = tanh_key.sigma_key
    select_lin_key = tanh_key.select_lin_key

    x_r_in = tanh_key.sigma_key.r_in
    from NssMPC import ArithmeticSecretSharing
    x_shift = ArithmeticSecretSharing(x_r_in) + x.flatten()
    x_shift = x_shift.restore()

    y_shift = x_shift // (x.scale // (2 ** table_scale_bit))
    y_shift.bit_len = x.bit_len - SCALE_BIT + table_scale_bit

    d_and_w = SigmaDICF.one_key_eval(
        [y_shift, y_shift + (2 ** (table_scale_bit + 1) - 1), y_shift - (2 ** (table_scale_bit + 1))], sigma_key,
        party.party_id)
    d = d_and_w[0]
    w = d_and_w[1] ^ d_and_w[2]

    d_and_w_b = RingTensor.cat([d, w], dim=0)
    d_and_w_a = b2a(d_and_w_b, party)
    d = d_and_w_a[:d.numel()]
    w = d_and_w_a[d.numel():]

    c = SelectLin.eval_with_comm(y_shift, w, d, select_lin_key)

    abs_tanh = LookUp.eval(c, tanh_key.look_up_key, tanh_key.look_up_table)
    sign = 2 * d - 1

    res = (abs_tanh * sign).reshape(shape)
    res.dtype = x.dtype
    return res
