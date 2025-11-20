#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
from NssMPC.infra.mpc.party import Party, PartyCtx
from NssMPC.primitives.secret_sharing.boolean import BooleanSecretSharing
from NssMPC.protocols.semi_honest_2pc.comparison import BooleanTriples


def beaver_and(x: BooleanSecretSharing, y: BooleanSecretSharing,
               party: Party = None) -> BooleanSecretSharing:
    """
    Perform bitwise AND operation for Boolean Secret Sharing (BSS) inputs using Beaver triples.

    This function uses Beaver triples (multiplication triples) to securely compute the AND of two shares
    ``x`` and ``y``. It handles inputs with different shapes by broadcasting the smaller input to match the larger one.

    Args:
        x: The first BSS input operand.
        y: The second BSS input operand.
        party: The party instance managing the communication. Defaults to None.

    Returns:
        BooleanSecretSharing: The secret-shared result of the AND operation.

    Examples:
        >>> res = beaver_and(x, y)
    """
    if party is None:
        party = PartyCtx.get()
    shape = x.shape if x.numel() > y.numel() else y.shape
    x = x.expand(shape).flatten()
    y = y.expand(shape).flatten()
    a, b, c = party.get_param(BooleanTriples, x.numel())
    a.dtype = b.dtype = c.dtype = x.dtype
    e = x ^ a
    f = y ^ b

    e_and_f = x.__class__.cat([e, f], dim=0)
    common_e_f = e_and_f.restore()
    length = common_e_f.shape[0] // 2
    common_e = common_e_f[:length]
    common_f = common_e_f[length:]

    res1 = common_e & common_f & party.party_id
    res2 = a.item & common_f
    res3 = common_e & b.item
    res = res1 ^ res2 ^ res3 ^ c.item

    res = x.__class__(res)
    return res.view(shape)
