#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.common.ring.ring_tensor import RingTensor


def rand(shape, party):
    """
    Get a random RSS(ReplicatedSecretSharing).

    .. note::
        What we get from this method is regarded as a SecretSharedRingPair.

    :param shape: The shape of the output RSS.
    :type shape: torch.Size or list[int]
    :param party: The party that hold the output.
    :type party: Party
    :return: A random RSS.
    :rtype: ReplicatedSecretSharing
    """

    num_of_value = 1
    for d in shape:
        num_of_value *= d
    r_0 = party.prg_0.random(num_of_value)
    r_1 = party.prg_1.random(num_of_value)
    from NssMPC.crypto.primitives.arithmetic_secret_sharing import ReplicatedSecretSharing
    r = ReplicatedSecretSharing([r_0, r_1])
    r = r.reshape(shape)
    return r


def rand_like(x, party):
    """
    Get a random RSS(ReplicatedSecretSharing) with specified shape.

    .. note::
        What we get from this method is regarded as a SecretSharedRingPair.

    :param x: The input from which we get the shape of the output.
    :type x: ReplicatedSecretSharing
    :param party: The party that hold the output.
    :type party: Party
    :return: A random RSS.
    :rtype: ReplicatedSecretSharing
    """
    r = rand(x.shape, party)
    # r = r.reshape(x.shape)
    if isinstance(x, RingTensor):
        r.dtype = x.dtype
    return r
