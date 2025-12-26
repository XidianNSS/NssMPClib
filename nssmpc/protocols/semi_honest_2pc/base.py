from nssmpc.infra.mpc.party import PartyCtx, Party
from nssmpc.infra.tensor import RingTensor
from nssmpc.primitives.secret_sharing import AdditiveSecretSharing


def sh2pc_recon(secret_share: AdditiveSecretSharing, target_id: int = None, party: Party = None) -> RingTensor | None:
    """Reconstruct the original data from secret shares.

    This method is used to reconstruct the original data by combining the shares
    held by different parties. It requires communication between the parties.

    Args:
        secret_share (AdditiveSecretSharing): The secret shared value.
        target_id (int): The party id the secret share restore to.
        party (Party, optional): The party context. Defaults to None.

    Returns:
        RingTensor: The reconstructed original data as a RingTensor.

    Examples:
        >>> res = sh2pc_recon(ass)
        >>> res = sh2pc_recon(ass, target_id=1)
    """
    if party is None:
        party = PartyCtx.get()
    if target_id is None:
        party.send(secret_share)
        other = party.recv()
        return secret_share.item + other.item
    assert target_id in [0, 1], "target_id must be 0 or 1"
    if target_id == party.party_id:
        return secret_share.item + party.recv().item
    else:
        party.send(secret_share)
        return None


def sh2pc_add_public_value(secret_share: AdditiveSecretSharing, public_value: RingTensor,
                           party: Party = None) -> AdditiveSecretSharing:
    """
    Adds a public value to a secret shared value in an Additive Secret Sharing scheme.

    Args:
        secret_share (AdditiveSecretSharing): The secret shared value.
        public_value (RingTensor): The public value to be added.
        party: The party performing the addition.

    Returns:
        AdditiveSecretSharing: The result of the addition as an AdditiveSecretSharing.

    Examples:
        >>> res = sh2pc_add_public_value(share, pub_val)
    """
    if party is None:
        party = PartyCtx.get()
    if party.party_id == 0:
        return AdditiveSecretSharing(secret_share.item + public_value)
    else:
        return AdditiveSecretSharing(secret_share.item + RingTensor.zeros_like(public_value))
