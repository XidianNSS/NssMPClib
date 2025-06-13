#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.common.utils.common_utils import list_rotate
from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.protocols.replicated_secret_sharing.semi_honest_functional.b2a import bit_injection


def secure_ge(x, y):
    """
    Securely compare the input `x` and `y` and return if x is greater than y.
    If x >= y, return the secret share of 1, else return the share of 0.

    :param x: The first in put of the comparison.
    :type x: ReplicatedSecretSharing
    :param y: The second in put of the comparison.
    :type y: ReplicatedSecretSharing
    :return: The result of comparison.
    :rtype: ReplicatedSecretSharing
    """
    z = x - y
    party = PartyRuntime.party

    if party.party_id == 2:
        out = RingTensor.zeros(z.shape, z.dtype, z.device)
        pass
    else:
        from NssMPC.crypto.primitives.function_secret_sharing.dicf import SigmaDICFKey
        k = PartyRuntime.party.get_param(SigmaDICFKey, z.numel())
        from NssMPC.config.configs import DEBUG_LEVEL
        if DEBUG_LEVEL:
            r_in = [k.r_in, RingTensor.convert_to_ring(0)]
        else:
            r_in = [k.r_in.reshape(z.shape), RingTensor.convert_to_ring(0)]

        r_in = list_rotate(r_in, party.party_id - 1)
        r = z + x.__class__(r_in)

        party.send((party.party_id + 1) % 2, r.item[party.party_id])
        r_other = party.receive((party.party_id + 1) % 2)
        z_shift = r.item[0] + r.item[1] + r_other

        z_shift.dtype = 'int'
        from NssMPC.crypto.primitives import SigmaDICF
        out = SigmaDICF.eval(z_shift, k, party.party_id)

    rand = x.__class__.rand_like(z, party)
    out = out ^ rand.item[0].get_bit(1) ^ rand.item[1].get_bit(1)
    party.send((party.party_id + 2) % 3, out)
    out_other = party.receive((party.party_id + 1) % 3)
    out = x.__class__([out, out_other])
    res = bit_injection(out)
    res.dtype = z.dtype
    return res * x.scale
