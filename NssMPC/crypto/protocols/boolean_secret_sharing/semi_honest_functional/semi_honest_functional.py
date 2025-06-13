#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.aux_parameter.beaver_triples import BooleanTriples


def beaver_and(x, y):
    """
    Perform bitwise AND operation for BSS inputs with beaver triples.

    This function uses Beaver triples to securely compute the AND of two BSS
    `x` and `y`. It handles BSS with different shapes by expanding the smaller BSS to match the larger one.

    :param x: The first BSS in the AND operation.
    :type x: BooleanSecretSharing
    :param y: The second BSS in the AND operation.
    :type y: BooleanSecretSharing
    :returns: The result of the Beaver's AND operation.
    :rtype: BooleanSecretSharing

    """
    shape = x.shape if x.numel() > y.numel() else y.shape
    x = x.expand(shape).flatten()
    y = y.expand(shape).flatten()
    a, b, c = PartyRuntime.party.get_param(BooleanTriples, x.numel())
    a.dtype = b.dtype = c.dtype = x.dtype
    e = x ^ a
    f = y ^ b

    e_and_f = x.__class__.cat([e, f], dim=0)
    common_e_f = e_and_f.restore()
    length = common_e_f.shape[0] // 2
    common_e = common_e_f[:length]
    common_f = common_e_f[length:]

    res1 = common_e & common_f & PartyRuntime.party.party_id
    res2 = a.item & common_f
    res3 = common_e & b.item
    res = res1 ^ res2 ^ res3 ^ c.item

    res = x.__class__(res)
    return res.view(shape)
