#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from typing import List, Union

import torch

from NssMPC.infra.tensor import RingTensor
from NssMPC.primitives.secret_sharing import ReplicatedSecretSharing


def rand(shape: Union[torch.Size, List[int]], party) -> ReplicatedSecretSharing:
    """Get a random RSS(ReplicatedSecretSharing).

    Note:
        What we get from this method is regarded as a SecretSharedRingPair.

    Args:
        shape: The shape of the output RSS.
        party (Party): The party that hold the output.

    Returns:
        A random RSS.

    Examples:
        >>> r = rand([2, 3], party)
    """

    num_of_value = 1
    for d in shape:
        num_of_value *= d
    r_0 = party.prg_0.random(num_of_value)
    r_1 = party.prg_1.random(num_of_value)
    r = ReplicatedSecretSharing([r_0, r_1])
    r = r.reshape(shape)
    return r


def rand_like(x: ReplicatedSecretSharing, party) -> ReplicatedSecretSharing:
    """Get a random RSS(ReplicatedSecretSharing) with specified shape.

    Note:
        What we get from this method is regarded as a SecretSharedRingPair.

    Args:
        x: The input from which we get the shape of the output.
        party (Party): The party that hold the output.

    Returns:
        A random RSS.

    Examples:
        >>> r = rand_like(x, party)
    """
    r = rand(x.shape, party)
    # r = r.reshape(x.shape)
    if isinstance(x, RingTensor):
        r.dtype = x.dtype
    return r
