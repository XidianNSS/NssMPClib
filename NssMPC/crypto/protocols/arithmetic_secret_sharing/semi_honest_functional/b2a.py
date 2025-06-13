#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
This document implements two methods for converting BoolSecretSharedTensor to ArithmeticSecretSharedTensor, namely the ``b2a``.
"""
import torch

from NssMPC import RingTensor
from NssMPC.config import DEVICE, data_type
from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.aux_parameter.b2a_keys.b2a_keys import B2AKey


def b2a(x: RingTensor, party):
    """
    Convert ``bool secret sharing`` input `x` to ``arithmetic secret sharing``.

    This method also achieves b2a, but with half the amount of communication compared to the method used in Sonic below.

    :param x: The bool secret sharing to be converted.
    :type x: RingTensor (BoolSecretSharedRingTensor, whose ring size is 2)
    :param party: The party who holds the secret sharing.
    :type party: Party
    :return: The ArithmeticSharedRingTensor after conversion.
    :rtype: RingTensor(ArithmeticSharedRingTensor)
    """
    with PartyRuntime(party):
        return crypten_b2a(x)


# todo: 输入应该为BSS
def sonic_b2a(x: RingTensor):
    """
    Convert ``bool secret sharing`` input `x` to ``arithmetic secret sharing``.

    This function is based on the secure B2A algorithm mentioned in `Sonic <https://ieeexplore.ieee.org/document/9674792>`_.
    It converses the bool secret sharing to arithmetic secret sharing.

    :param x: The bool secret sharing to be converted.
    :type x: RingTensor (BoolSecretSharedRingTensor, whose ring size is 2)
    :return: The ArithmeticSharedRingTensor after conversion.
    :rtype: RingTensor(ArithmeticSharedRingTensor)
    """
    party = PartyRuntime.party
    from NssMPC.crypto.primitives.arithmetic_secret_sharing import ArithmeticSecretSharing
    zero = RingTensor.zeros(x.shape, 'int', device=DEVICE)
    if party.party_id == 0:
        a = ArithmeticSecretSharing(x)
        b = ArithmeticSecretSharing(zero)
    else:
        b = ArithmeticSecretSharing(x)
        a = ArithmeticSecretSharing(zero)
    return a + b - a * b * 2


def crypten_b2a(x: RingTensor):
    """
    Convert ``bool secret sharing`` input `x` to ``arithmetic secret sharing``.

    This method also achieves b2a, but with half the amount of communication compared to the method used in Sonic above.

    :param x: The bool secret sharing to be converted.
    :type x: RingTensor (BoolSecretSharedRingTensor, whose ring size is 2)
    :return: The ArithmeticSharedRingTensor after conversion.
    :rtype: RingTensor(ArithmeticSharedRingTensor)
    """
    party = PartyRuntime.party
    shape = x.shape
    x = x.flatten()
    b2a_key = party.get_param(B2AKey, x.numel())
    r = b2a_key.r
    from NssMPC.crypto.primitives.arithmetic_secret_sharing import ArithmeticSecretSharing
    from NssMPC.crypto.primitives.boolean_secret_sharing.boolean_secret_sharing import BooleanSecretSharing
    x_shift = BooleanSecretSharing((x + b2a_key.r.item) % 2)
    x_shift.item = x_shift.item.to(torch.bool)

    x_shift = x_shift.restore()
    x_shift.tensor = x_shift.tensor.to(data_type)

    return (ArithmeticSecretSharing(x_shift * party.party_id) + r - r * x_shift * 2).reshape(shape)
