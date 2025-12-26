#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
from typing import Optional, Union, List

import torch

from nssmpc.infra.mpc.party import Party, PartyCtx
from nssmpc.infra.tensor import RingTensor
from nssmpc.primitives.secret_sharing import ReplicatedSecretSharing


def hm3pc_open(x: ReplicatedSecretSharing, party: Party | None = None) -> RingTensor:
    """
    Reconstructs the secret from shares to all parties.

    This function gathers shares from other parties to reconstruct the original secret value.
    It includes a check to ensure consistency between shares received from different parties
    to detect malicious behavior.

    Args:
        x: The secret shares to be opened.
        party: The party instance. If None, uses the current party context.

    Returns:
        RingTensor: The reconstructed secret value.

    Raises:
        ValueError: If the shares are inconsistent (potential malicious party).

    Examples:
        >>> x_recon = hm3pc_open(x_share)
    """
    if party is None:
        party = PartyCtx.get()
    # send x0 to P_{i+1}
    party.send((party.party_id + 1) % 3, x.item[0])
    # receive x2 from P_{i+1}
    x_2_0 = party.recv((party.party_id + 2) % 3)
    # send x1 to P_{i-1}
    party.send((party.party_id + 2) % 3, x.item[1])
    # receive x2 from P_{i-1}
    x_2_1 = party.recv((party.party_id + 1) % 3)

    cmp = x_2_0 - x_2_1
    # print(cmp)
    cmp = cmp.tensor.flatten().sum(axis=0)
    if cmp != 0:
        raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")

    return x.item[0] + x.item[1] + x_2_0


def hm3pc_coin(num_of_value: int | list, party) -> ReplicatedSecretSharing:
    """
    Generates a random public value shared by all parties.

    This function generates random shares and then opens them to establish a common
    random value known to all parties.

    Args:
        num_of_value: The number of random values to generate, or a list defining the shape.
        party: The party instance.

    Returns:
        ReplicatedSecretSharing: The generated random value.

    Examples:
        >>> r = hm3pc_coin(10, party)
    """
    rss_r = rand([num_of_value], party)
    r = hm3pc_open(rss_r)
    return r


def hm3pc_recv_share_from(input_id: int, party: Party) -> ReplicatedSecretSharing:
    """
    Receives a secret shared by another party.

    This function is used by the receiving parties when a specific party shares a value.
    It involves receiving the shape, generating random shares, and verifying consistency.

    Args:
        input_id: The ID of the party providing the input.
        party: The party instance.

    Returns:
        ReplicatedSecretSharing: The received secret shares.

    Examples:
        >>> x_share = hm3pc_recv_share_from(input_id=0, party=party)
    """
    # receive shape from P_i
    shape_of_received_shares = party.recv(input_id)
    r = rand(shape_of_received_shares, party)
    _ = hm3pc_recon(r, input_id)

    # receive delta from P_{input_id}
    delta = party.recv(input_id)

    # check if delta is same
    other_id = (0 + 1 + 2) - party.party_id - input_id
    # send delta to P_{other_id}
    party.send(other_id, delta)
    delta_other = party.recv(other_id)

    cmp = delta - delta_other
    cmp = cmp.tensor.flatten().sum(axis=0)
    if cmp != 0:
        raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")
    res = r + delta
    res.dtype = delta.dtype
    return res


def hm3pc_share_and_send(x: RingTensor, party=None) -> ReplicatedSecretSharing:
    """
    Shares a secret value from the current party to all parties.

    The current party acts as the input party, distributing shares of its secret `x`
    to the other parties.

    Args:
        x: The secret value to be shared.
        party: The party instance.

    Returns:
        ReplicatedSecretSharing: The shared secret.

    Raises:
        TypeError: If `x` is not a RingTensor.

    Examples:
        >>> x_share = hm3pc_share_and_send(x, party)
    """
    if not isinstance(x, RingTensor):
        raise TypeError("unsupported data type(s) ")
    if party is None:
        party = PartyCtx.get()

    shape = x.shape
    # send shape to P_{i+1} and P_{i-1}
    party.send((party.party_id + 1) % 3, shape)
    party.send((party.party_id + 2) % 3, shape)
    r = rand_like(x, party)
    r_recon = hm3pc_recon(r, party.party_id)
    delta = x - r_recon
    # broadcasts delta to all parties
    party.send((party.party_id + 1) % 3, delta)
    party.send((party.party_id + 2) % 3, delta)
    res = r + delta
    return res


def hm3pc_recon(x: ReplicatedSecretSharing, target_id: int, party: Party = None) -> Optional[ReplicatedSecretSharing]:
    """
    Reconstructs the secret to a specific target party.

    Only the target party will receive the necessary shares to reconstruct the secret.
    Other parties assist in the process but do not learn the secret.

    Args:
        x: The secret shares to be reconstructed.
        target_id: The ID of the party that will receive the reconstructed secret.
        party: The party instance. If None, uses the current party context.

    Returns:
        Optional[ReplicatedSecretSharing]: The reconstructed secret for the target party, or None for other parties.

    Examples:
        >>> x_val = hm3pc_recon(x_share, target_id=0)
    """
    if party is None:
        party = PartyCtx.get()
    # P_{i+1} send x_{i+2} to P_{i}
    if party.party_id == (target_id + 1) % 3:
        party.send(target_id, x.item[1])
    # P_{i-1} send x_{i+2} to P_{i}
    elif party.party_id == (target_id + 2) % 3:
        party.send(target_id, x.item[0])

    elif party.party_id == target_id:
        x_2_0 = party.recv((party.party_id + 2) % 3)
        x_2_1 = party.recv((party.party_id + 1) % 3)

        cmp = x_2_0 - x_2_1
        cmp = cmp.tensor.flatten().sum(axis=0)
        if cmp != 0:
            # raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")
            pass
        if party.party_id == target_id:
            return x.item[0] + x.item[1] + x_2_0
    return None


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
