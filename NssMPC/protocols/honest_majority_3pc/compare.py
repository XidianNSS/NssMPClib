#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
from NssMPC.primitives import ReplicatedSecretSharing
from NssMPC.protocols.honest_majority_3pc.mac_check import MAC_BUFFER


def secure_ge(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing) -> ReplicatedSecretSharing:
    """
    Securely computes whether x is greater than or equal to y (x >= y).

    This function performs a secure comparison between two ReplicatedSecretSharing values.
    It calculates the difference z = x - y and then determines the Most Significant Bit (MSB)
    of z to decide the result.

    Args:
        x: The left operand of the comparison.
        y: The right operand of the comparison.

    Returns:
        ReplicatedSecretSharing: A sharing of 1 if x >= y, otherwise a sharing of 0.

    Examples:
        >>> res = secure_ge(x, y)
    """
    z = x - y
    dtype = x.dtype
    x.dtype = y.dtype = 'int'

    shape = z.shape
    from NssMPC.protocols.honest_majority_3pc.msb_with_os import \
        msb_with_os_without_mac_check
    res, mac_v, mac_key = msb_with_os_without_mac_check(z)

    MAC_BUFFER.add(res, mac_v, mac_key)
    # mac_check(res, mac_v, mac_key)

    res.dtype = dtype
    return (res.view(shape) * res.scale - 1) * -1
