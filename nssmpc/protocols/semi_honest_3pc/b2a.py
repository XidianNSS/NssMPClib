#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
from typing import Optional

from nssmpc.infra.mpc.party import Party, PartyCtx
from nssmpc.infra.tensor.ring_tensor import RingTensor
from nssmpc.primitives.oblivious_transfer.ot_aby3 import OT
from nssmpc.primitives.secret_sharing import ReplicatedSecretSharing


def sh3pc_bit_injection(x: ReplicatedSecretSharing, party: Party = None) -> Optional[ReplicatedSecretSharing]:
    """Implements the single-bit B2A protocol (Bit Injection).

    Based on ABY3, where P2 is the sender, P1 is the receiver, and P0 is the helper.
    It converts a boolean sharing of a bit into an arithmetic sharing of the same bit.

    Args:
        x: The input boolean secret sharing (representing bits) to be converted.
        party: The party instance managing the communication. Defaults to None.

    Returns:
        The arithmetic secret sharing of the input bit(s).

    Examples:
        >>> res = sh3pc_bit_injection(x)
    """
    if party is None:
        party = PartyCtx.get()
    if party.party_id == 0:
        c0 = RingTensor.random(x.shape, dtype=x.dtype, device=x.device)
        party.send(2, c0)
        OT.helper(x.item[1], party, 2, 1)
        c1 = party.recv(1)
        return x.__class__([c0, c1])

    if party.party_id == 1:
        c2 = RingTensor.random(x.shape, dtype=x.dtype, device=x.device)
        party.send(2, c2)
        c1 = OT.receiver(x.item[0], party, 2, 0)
        party.send(0, c1)
        return x.__class__([c1, c2])

    if party.party_id == 2:
        c0 = party.recv(0)
        c2 = party.recv(1)
        m0 = (x.item[0] ^ x.item[1]) - c0 - c2
        m1 = (x.item[0] ^ x.item[1] ^ 1) - c0 - c2
        OT.sender(m0, m1, party, 1, 0)
        return x.__class__([c2, c0])
