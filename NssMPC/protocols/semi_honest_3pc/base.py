from NssMPC.infra.tensor import RingTensor
from NssMPC.infra.mpc.party import PartyCtx, Party
from NssMPC.primitives.secret_sharing import ReplicatedSecretSharing


def add_public_value(secret_share:ReplicatedSecretSharing, public_value:RingTensor, party: Party = None) -> ReplicatedSecretSharing:
    """Adds a public value to a secret shared value.

    Args:
        secret_share: The secret shared value.
        public_value: The public value to add.
        party: The party instance managing the communication. Defaults to None.

    Returns:
        The result of adding the public value to the secret share.

    Examples:
        >>> res = add_public_value(x, pub_val)
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