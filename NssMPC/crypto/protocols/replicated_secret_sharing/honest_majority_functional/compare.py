#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.config.runtime import MAC_BUFFER


def secure_ge(x, y):
    """
    Gets the most significant bit of an RSS sharing ⟨x⟩
    :param x: an RSS sharing ⟨x⟩
    :param y: an RSS sharing ⟨y⟩
    :return: the most significant bit of x
    """
    z = x - y
    dtype = x.dtype
    x.dtype = y.dtype = 'int'

    shape = z.shape
    from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.msb_with_os import \
        msb_with_os_without_mac_check
    res, mac_v, mac_key = msb_with_os_without_mac_check(z)

    MAC_BUFFER.add(res, mac_v, mac_key)
    # mac_check(res, mac_v, mac_key)

    res.dtype = dtype
    return (res.view(shape) * res.scale - 1) * -1
