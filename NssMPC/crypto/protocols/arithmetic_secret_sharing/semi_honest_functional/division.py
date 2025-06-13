#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional import b2a
from NssMPC.crypto.primitives.function_secret_sharing.dicf import SigmaDICF
from NssMPC.config import SCALE_BIT
from NssMPC.crypto.aux_parameter.look_up_table_keys.div_key import DivKey
from NssMPC.crypto.protocols.look_up_table import LookUp


def secure_div(x, y):
    """
    Implement ASS division protocols using iterative method
    Correct only if y > 0 and y < 2 ** 2f
    TODO： support y in other range
    This method performs division of two ASS using the secure_div protocol, which will insure the security of the computation.
    The main idea behind this method is based on the *Goldschmidt approximate division*.

    .. note::
        Only if *y > 0* and *y< 2 ** 2f* the result is correct (*f* refers to the scale bit of y).

    :param x: The dividend.
    :type x: ArithmeticSecretSharing
    :param y: The divisor.
    :type y: ArithmeticSecretSharing
    :return: The result of division.
    :rtype: ArithmeticSecretSharing

    """
    if x.numel() > y.numel():
        return x * secure_inv(y)
    else:
        neg_exp2_k = get_neg_exp2_k(y)
        a = x * neg_exp2_k
        b = y * neg_exp2_k
        w = b * (-2) + 2.9142
        e0 = -(b * w) + 1
        e1 = e0 * e0

        return a * w * (e0 + 1) * (e1 + 1)


def secure_inv(x):
    neg_exp2_k = get_neg_exp2_k(x)
    b = x * neg_exp2_k
    w = b * (-2) + 2.9142
    e0 = -(b * w) + 1
    e1 = e0 * e0

    return w * neg_exp2_k * (e0 + 1) * (e1 + 1)


def get_neg_exp2_k(divisor):
    div_key = PartyRuntime.party.get_param(DivKey, divisor.numel())
    sigma_key = div_key.sigma_key
    nexp2_key = div_key.neg_exp2_key

    y_shape = divisor.shape
    y_shift = divisor.__class__(sigma_key.r_in) + divisor.flatten()
    y_shift = y_shift.restore()
    y_shift = y_shift.view(y_shape)

    y_minus_powers = [y_shift - (2 ** i) for i in range(1, 2 * SCALE_BIT + 1)]
    k = SigmaDICF.one_key_eval(y_minus_powers, sigma_key, PartyRuntime.party.party_id)
    k = b2a(k, PartyRuntime.party).sum(dim=0)
    return LookUp.eval(k + 1, nexp2_key.look_up_key, nexp2_key.table)
