from NssMPC.infra.mpc.party import PartyCtx, Party
from NssMPC.infra.tensor import RingTensor
from NssMPC.primitives.secret_sharing import AdditiveSecretSharing


def add_public_value(secret_share, public_value, party: Party = None) -> AdditiveSecretSharing:
    """
    Adds a public value to a secret shared value in an Additive Secret Sharing scheme.

    Args:
        secret_share (AdditiveSecretSharing): The secret shared value.
        public_value (RingTensor): The public value to be added.
        party: The party performing the addition.

    Returns:
        AdditiveSecretSharing: The result of the addition as an AdditiveSecretSharing.

    Examples:
        >>> res = add_public_value(share, pub_val)
    """
    if party is None:
        party = PartyCtx.get()
    if party.party_id == 0:
        return AdditiveSecretSharing(secret_share.item + public_value)
    else:
        return AdditiveSecretSharing(secret_share.item + RingTensor.zeros_like(public_value))