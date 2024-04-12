from common.tensor import *


def b2a(x: RingTensor, party):
    """
    This function is based on the secure B2A algorithm mentioned in sonic.
    It converses the bool secret sharing to arithmetic secret sharing.

    Args:
        x (RingTensor): bool secret sharing
        party: the party who hold the secret sharing

    Returns:
        ArithmeticSharedRingTensor
    """
    from crypto.primitives.arithmetic_secret_sharing.arithmetic_shared_ring_tensor import ArithmeticSharedRingTensor
    zero = RingFunc.zeros(x.shape, 'int', device=DEVICE)
    if party.party_id == 0:
        a = ArithmeticSharedRingTensor(x, party)
        b = ArithmeticSharedRingTensor(zero, party)
    else:
        b = ArithmeticSharedRingTensor(x, party)
        a = ArithmeticSharedRingTensor(zero, party)
    return a + b - a * b * 2
