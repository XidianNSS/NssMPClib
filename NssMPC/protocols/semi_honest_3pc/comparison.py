#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
from NssMPC.config import DEBUG_LEVEL
from NssMPC.infra.mpc.party import Party, PartyCtx
from NssMPC.infra.tensor.ring_tensor import RingTensor
from NssMPC.infra.utils.common_utils import list_rotate
from NssMPC.primitives.secret_sharing import ReplicatedSecretSharing
from NssMPC.primitives.secret_sharing.function import SigmaDICF, SigmaDICFKey
from NssMPC.protocols.semi_honest_3pc.b2a import bit_injection


def secure_ge(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing, party: Party = None) -> ReplicatedSecretSharing:
    """Securely compare inputs x and y to check if x >= y.

    If x >= y, returns a secret share of 1, otherwise returns a share of 0.

    Args:
        x: The first input of the comparison (LHS).
        y: The second input of the comparison (RHS).
        party: The party instance managing the communication. Defaults to None.

    Returns:
        The result of the comparison (1 if x >= y else 0).

    Examples:
        >>> res = secure_ge(x, y)
    """
    if party is None:
        party = PartyCtx.get()

    z = x - y

    if party.party_id == 2:
        out = RingTensor.zeros(z.shape, z.dtype, z.device)
        pass
    else:
        k = party.get_param(SigmaDICFKey, z.numel())
        if DEBUG_LEVEL:
            r_in = [k.r_in, RingTensor.convert_to_ring(0)]
        else:
            r_in = [k.r_in.reshape(z.shape), RingTensor.convert_to_ring(0)]

        r_in = list_rotate(r_in, party.party_id - 1)
        r = z + x.__class__(r_in)

        party.send((party.party_id + 1) % 2, r.item[party.party_id])
        r_other = party.recv((party.party_id + 1) % 2)
        z_shift = r.item[0] + r.item[1] + r_other

        z_shift.dtype = 'int'
        out = SigmaDICF.eval(z_shift, k, party.party_id)

    rand = x.__class__.rand_like(z, party)
    out = out ^ rand.item[0].get_bit(1) ^ rand.item[1].get_bit(1)
    party.send((party.party_id + 2) % 3, out)
    out_other = party.recv((party.party_id + 1) % 3)
    out = x.__class__([out, out_other])
    res = bit_injection(out)
    res.dtype = z.dtype
    return res * x.scale
