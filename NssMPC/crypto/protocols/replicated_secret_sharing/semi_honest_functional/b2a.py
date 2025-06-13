#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.primitives.oblivious_transfer.ot_aby3 import OT


def bit_injection(x):
    """
    Implements the single-bit b2a protocol.
    Based on `ABY3 <https://eprint.iacr.org/2018/403.pdf>`_, where P2 is the sender, P1 is the receiver, and P0 is the helper.


    :param x: The input `x` that needs to undergo b2a conversion.
    :type x: ReplicatedSecretSharing
    :return: The result after converting the input using b2a.
    :rtype: ReplicatedSecretSharing
    """
    party = PartyRuntime.party
    if party.party_id == 0:
        c0 = RingTensor.random(x.shape, dtype=x.dtype, device=x.device)
        party.send(2, c0)
        OT.helper(x.item[1], party, 2, 1)
        c1 = party.receive(1)
        return x.__class__([c0, c1])

    if party.party_id == 1:
        c2 = RingTensor.random(x.shape, dtype=x.dtype, device=x.device)
        party.send(2, c2)
        c1 = OT.receiver(x.item[0], party, 2, 0)
        party.send(0, c1)
        return x.__class__([c1, c2])

    if party.party_id == 2:
        c0 = party.receive(0)
        c2 = party.receive(1)
        m0 = (x.item[0] ^ x.item[1]) - c0 - c2
        m1 = (x.item[0] ^ x.item[1] ^ 1) - c0 - c2
        OT.sender(m0, m1, party, 1, 0)
        return x.__class__([c2, c0])
