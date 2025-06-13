#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.config.runtime import PartyRuntime


def mul_with_out_trunc(x, y):
    """
    Implement the multiplication of RSS without truncation.

    :param x: One of the multipliers
    :type x: ReplicatedSecretSharing
    :param y: The other multiplier.
    :type y: ReplicatedSecretSharing
    :return: The shared value of the result of the multiplication。
    :rtype: ReplicatedSecretSharing
    """
    z_shared = RingTensor.mul(x.item[0], y.item[0]) + RingTensor.mul(x.item[0], y.item[1]) + RingTensor.mul(x.item[1],
                                                                                                            y.item[0])
    from NssMPC.crypto.primitives.arithmetic_secret_sharing import ReplicatedSecretSharing
    result = ReplicatedSecretSharing.reshare(z_shared, PartyRuntime.party)
    return result


def matmul_with_out_trunc(x, y):
    """
    Implement the matrix multiplication of RSS without truncation.

    :param x: One of the matrix multipliers
    :type x: ReplicatedSecretSharing
    :param y: The other matrix multiplier.
    :type y: ReplicatedSecretSharing
    :return: The shared value of the result of the matrix multiplication。
    :rtype: ReplicatedSecretSharing
    """
    t_i = \
        RingTensor.matmul(x.item[0], y.item[0]) + \
        RingTensor.matmul(x.item[0], y.item[1]) + \
        RingTensor.matmul(x.item[1], y.item[0])
    from NssMPC.crypto.primitives.arithmetic_secret_sharing import ReplicatedSecretSharing
    result = ReplicatedSecretSharing.reshare(t_i, PartyRuntime.party)
    return result
