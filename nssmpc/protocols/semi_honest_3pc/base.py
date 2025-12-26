from nssmpc import Party3PC
from nssmpc.infra.mpc import PartyCtx

from nssmpc.infra.tensor import RingTensor
from nssmpc.primitives.secret_sharing import ReplicatedSecretSharing


def sh3pc_recon(secret_share: ReplicatedSecretSharing, target_id: int = None, party: Party3PC = None):
    """Reconstruct the original data from the secret shares.

    Perform the send-and-receive process to reconstruct the original RingTensor.

    Args:
        secret_share (ReplicatedSecretSharing): The secret shared value.
        target_id (int, optional): The id of the party that holds the original data. Defaults to None.
        party (Party3PC, optional): Current party. Defaults to None.

    Returns:
        RingTensor: The restored RingTensor.

    Examples:
        >>> res = rss.recon()
    """
    # 发送部分
    if party is None:
        party = PartyCtx.get()
    if target_id is None:
        party.send((party.party_id + 1) % 3, secret_share.item[0])
        other = party.recv((party.party_id + 2) % 3)
        return secret_share.item[0] + secret_share.item[1] + other
    assert target_id in [0, 1, 2], 'target_id must be 0, 1 or 2'
    if party.party_id == target_id:
        return secret_share.item[0] + secret_share.item[1] + party.recv((party.party_id + 2) % 3)
    elif party.party_id == (target_id + 2) % 3:
        party.send(target_id, secret_share.item[0])
    return None


def sh3pc_add_public_value(secret_share: ReplicatedSecretSharing, public_value: RingTensor,
                           party: Party3PC = None) -> ReplicatedSecretSharing:
    """Adds a public value to a secret shared value.

    Args:
        secret_share: The secret shared value.
        public_value: The public value to add.
        party: The party instance managing the communication. Defaults to None.

    Returns:
        The result of adding the public value to the secret share.

    Examples:
        >>> res = sh3pc_add_public_value(x, pub_val)
    """
    if party is None:
        party = PartyCtx.get()
    zeros = RingTensor.zeros_like(public_value, dtype=public_value.dtype, device=public_value.device)
    if party.party_id == 0:
        return ReplicatedSecretSharing([secret_share.item[0] + public_value, secret_share.item[1] + zeros])
    elif party.party_id == 2:
        return ReplicatedSecretSharing([secret_share.item[0] + zeros, secret_share.item[1] + public_value])
    else:
        return ReplicatedSecretSharing([secret_share.item[0] + zeros, secret_share.item[1] + zeros])
