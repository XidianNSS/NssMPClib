from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.crypto.primitives.oblivious_transfer.ot_aby3 import OT


def bit_injection(x):
    """
    单bit位b2a协议
    参考ABY3实现，令P2为sender，P1为receiver，P0为helper

    :param x:
    :return:
    """
    party = x.party
    if party.party_id == 0:
        c0 = RingTensor.random(x.shape, dtype=x.dtype, device=x.device)
        party.send(2, c0)
        OT.helper(x.item[1], party, 2, 1)
        c1 = party.receive(1)
        return x.__class__([c0, c1], party)

    if party.party_id == 1:
        c2 = RingTensor.random(x.shape, dtype=x.dtype, device=x.device)
        party.send(2, c2)
        c1 = OT.receiver(x.item[0], party, 2, 0)
        party.send(0, c1)
        return x.__class__([c1, c2], party)

    if party.party_id == 2:
        c0 = party.receive(0)
        c2 = party.receive(1)
        m0 = (x.item[0] ^ x.item[1]) - c0 - c2
        m1 = (x.item[0] ^ x.item[1] ^ 1) - c0 - c2
        OT.sender(m0, m1, party, 1, 0)
        return x.__class__([c2, c0], party)
