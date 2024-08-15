import torch
from NssMPC import RingTensor
from NssMPC.config import DEVICE, data_type
from NssMPC.crypto.aux_parameter.b2a_keys.b2a_keys import B2AKey


def b2a(x: RingTensor, party):
    return x_b2a(x, party)


# todo: 输入应该为BSS
def sonic_b2a(x: RingTensor, party):
    """
    This function is based on the secure B2A algorithm mentioned in sonic.
    It converses the bool secret sharing to arithmetic secret sharing.

    Args:
        x (RingTensor): bool secret sharing
        party: the party who hold the secret sharing

    Returns:
        ArithmeticSharedRingTensor
    """
    from NssMPC.crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing
    zero = RingTensor.zeros(x.shape, 'int', device=DEVICE)
    if party.party_id == 0:
        a = ArithmeticSecretSharing(x, party)
        b = ArithmeticSecretSharing(zero, party)
    else:
        b = ArithmeticSecretSharing(x, party)
        a = ArithmeticSecretSharing(zero, party)
    return a + b - a * b * 2


def x_b2a(x: RingTensor, party):
    shape = x.shape
    x = x.flatten()
    b2a_key = party.get_param(B2AKey, x.numel())
    r = b2a_key.r
    r.party = party
    from NssMPC.crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing
    from NssMPC.crypto.primitives.boolean_secret_sharing.boolean_secret_sharing import BooleanSecretSharing
    x_shift = BooleanSecretSharing((x + b2a_key.r.item) % 2, party)
    x_shift.item = x_shift.item.to(torch.bool)

    x_shift = x_shift.restore()
    x_shift.tensor = x_shift.tensor.to(data_type)

    return (ArithmeticSecretSharing(x_shift * party.party_id, party) + r - r * x_shift * 2).reshape(shape)
