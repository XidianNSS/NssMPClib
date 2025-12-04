import torch

from NssMPC import Party2PC, Party3PC
from NssMPC.infra.mpc import PartyCtx
from NssMPC.infra.tensor import RingTensor
from NssMPC.primitives.secret_sharing.arithmetic import SecretSharingScheme, AdditiveSecretSharing, \
    ReplicatedSecretSharing


def SecretTensor(tensor: torch.Tensor = None,
                 src_id: int = None) -> AdditiveSecretSharing | ReplicatedSecretSharing | None:
    """Creates an arithmetic secret shared tensor.

    Args:
        tensor: The input tensor to be secret shared. If `src` is specified, this argument is ignored.
        src_id: The source party ID from which to receive the secret shared tensor. If `None`, the tensor is secret shared from the local party.

    Returns:
        SecretSharingScheme: An arithmetic secret shared tensor.
    """
    party = PartyCtx.get()
    if tensor is None and src_id is None:
        raise ValueError("Either `tensor` or `src` must be provided.")
    if isinstance(party, Party2PC):
        if tensor is not None:
            share_0, share_1 = AdditiveSecretSharing.share(RingTensor.convert_to_ring(tensor))
            party.send(share_1)
            return share_0
        else:
            return party.recv()
    elif isinstance(party, Party3PC):
        if tensor is not None:
            share_0, share_1, share_2 = ReplicatedSecretSharing.share(RingTensor.convert_to_ring(tensor))
            party.send((party.party_id + 1) % 3, share_1)
            party.send((party.party_id + 2) % 3, share_2)
            return share_0
        elif src_id is not None:
            return party.recv(src_id)
    else:
        raise RuntimeError("Unsupported party type for SecretTensor.")
