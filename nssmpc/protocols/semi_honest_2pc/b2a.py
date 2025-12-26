#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
This document implements two methods for converting BoolSecretSharedTensor to ArithmeticSecretSharedTensor, namely the ``b2a``.
"""
from typing import Tuple

import torch

from nssmpc.config import DEVICE, data_type
from nssmpc.infra.mpc.aux_parameter.parameter import Parameter
from nssmpc.infra.mpc.party import PartyCtx, Party
from nssmpc.infra.tensor import RingTensor
from nssmpc.primitives.secret_sharing import AdditiveSecretSharing


def sh2pc_b2a(x: RingTensor, party: Party) -> AdditiveSecretSharing:
    """Converts a boolean secret shared tensor to an arithmetic secret shared tensor.

    Args:
        x: Boolean secret shared input. Elements must be 0 or 1.
        party: The party instance.

    Returns:
        AdditiveSecretSharing: Arithmetic secret shared tensor.

    Examples:
        >>> res = sh2pc_b2a(x, party)
    """
    return sh2pc_b2a_crypten(x, party)


# todo: 输入应该为BSS
def sh2pc_b2a_sonic(x: RingTensor, party: Party | None = None) -> AdditiveSecretSharing:
    """Converts boolean shares to arithmetic shares using the Sonic protocol.

    Args:
        x: Boolean secret shared input. Elements must be 0 or 1.
        party: The party instance. Defaults to current context.

    Returns:
        AdditiveSecretSharing: Arithmetic secret shared tensor.

    Examples:
        >>> res = sh2pc_b2a_sonic(x)
    """
    if party is None:
        party = PartyCtx.get()
    zero = RingTensor.zeros(x.shape, 'int', device=DEVICE)
    if party.party_id == 0:
        a = AdditiveSecretSharing(x)
        b = AdditiveSecretSharing(zero)
    else:
        b = AdditiveSecretSharing(x)
        a = AdditiveSecretSharing(zero)
    return a + b - a * b * 2


def sh2pc_b2a_crypten(x: RingTensor, party: Party = None) -> AdditiveSecretSharing:
    """Converts boolean shares to arithmetic shares using an optimized protocol.

    Args:
        x: Boolean secret shared input. Elements must be 0 or 1.
        party: The party instance. Defaults to current context.

    Returns:
        AdditiveSecretSharing: Arithmetic secret shared tensor.

    Examples:
        >>> res = sh2pc_b2a_crypten(x)
    """
    if party is None:
        party = PartyCtx.get()
    shape = x.shape
    x = x.flatten()
    b2a_key = party.get_param(B2AKey, x.numel())
    r = b2a_key.r
    from nssmpc.primitives.secret_sharing import AdditiveSecretSharing
    from nssmpc.primitives.secret_sharing.boolean import BooleanSecretSharing
    x_shift = BooleanSecretSharing((x + b2a_key.r.item) % 2)
    x_shift.item = x_shift.item.to(torch.bool)

    x_shift = x_shift.recon(party)
    x_shift.tensor = x_shift.tensor.to(data_type)

    return (AdditiveSecretSharing(x_shift * party.party_id) + r - r * x_shift * 2).reshape(shape)


class B2AKey(Parameter):
    """Parameter class for B2A conversion keys.

    Attributes:
        r (AdditiveSecretSharing): Random number shares used for obfuscation.
    """

    def __init__(self):
        """Initializes a B2AKey instance."""
        self.r = None

    @staticmethod
    def gen(num_of_params: int) -> Tuple[Parameter, Parameter]:
        """Generates a pair of B2AKey parameters.

        Args:
            num_of_params: Number of keys to generate.

        Returns:
            Tuple[Parameter, Parameter]: Key shares for party 0 and party 1.

        Examples:
            >>> k0, k1 = B2AKey.gen(10)
        """
        r = RingTensor.random([num_of_params], down_bound=0, upper_bound=2)
        k0, k1 = B2AKey(), B2AKey()
        from nssmpc.primitives.secret_sharing import AdditiveSecretSharing
        k0.r, k1.r = AdditiveSecretSharing.share(r, 2)

        return k0, k1
