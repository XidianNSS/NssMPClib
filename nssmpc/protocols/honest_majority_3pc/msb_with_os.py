#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
The implementation is based on the work of `Fu J, Cheng K, Xia Y, et al. Private decision tree evaluation with malicious security via function secret sharing 2024: 310-330`.
For reference, see the `paper <https://link.springer.com/chapter/10.1007/978-3-031-70890-9_16>`_.
"""
from nssmpc.config import DEVICE
from nssmpc.infra.mpc.aux_parameter.parameter import Parameter
from nssmpc.infra.mpc.party import PartyCtx
from nssmpc.infra.mpc.party.party import Party3PC
from nssmpc.infra.tensor import RingTensor
from nssmpc.primitives.secret_sharing import AdditiveSecretSharing, ReplicatedSecretSharing
from nssmpc.primitives.secret_sharing.function import VSigma, VSigmaKey
from nssmpc.protocols.semi_honest_2pc.b2a import sh2pc_b2a
from nssmpc.protocols.semi_honest_2pc.base import sh2pc_recon


def hm3pc_msb_with_os_without_mac_check(x: ReplicatedSecretSharing, share_table=None, party: Party3PC = None):
    """
    Computes the Most Significant Bit (MSB) of a secret sharing value using Oblivious Selection (OS) without performing a MAC check immediately.

    This protocol is based on the work of Fu et al. (2024) for private decision tree evaluation.
    It involves retrieving pre-generated parameters (VSigmaKey), performing local computations,
    reconstructing masked values, evaluating VSigma functions, and converting boolean shares to arithmetic shares.

    Args:
        x: The input secret sharing value for which to compute the MSB.
        share_table (tuple, optional): A tuple containing share table and smac table. If None, it is retrieved from parameters. Defaults to None.
        party: The party instance participating in the protocol. If None, the current party context is used. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - v (ReplicatedSecretSharing): The secret shared MSB of the input x.
            - mac_v (ReplicatedSecretSharing): The MAC share of the result v.
            - mac_key (ReplicatedSecretSharing or None): The MAC key share used for verification. None if share_table was provided.

    Raises:
        Exception: If the party ID is invalid.

    Examples:
        >>> v, mac_v, mac_key = hm3pc_msb_with_os_without_mac_check(x)
    """
    if party is None:
        party = PartyCtx.get()
    shape = x.shape
    x = x.reshape(-1, 1)
    num = x.shape[0]

    previous_party = party.virtual_party_with_previous
    next_party = party.virtual_party_with_next
    previous_rdx, previous_key, previous_c = previous_party.get_param(
        VSigmaKey, num, tag=f'{party.party_id}'
    )
    next_rdx, next_key, next_c = next_party.get_param(
        VSigmaKey, num, tag=f'{(party.party_id + 1) % 3}'
    )

    def match_shape(value):
        if value.numel() == 1:
            return value.expand(*x.shape)
        return value.reshape(x.shape)

    def evaluate_previous_pair():
        local_delta = x.item[1] + match_shape(previous_rdx.item)
        delta = sh2pc_recon(AdditiveSecretSharing(local_delta), party=previous_party)
        boolean_share, _ = VSigma.eval(
            delta,
            (previous_key.to(DEVICE), previous_c.to(DEVICE)),
            previous_party.party_id,
        )
        return sh2pc_b2a(boolean_share, previous_party)

    def evaluate_next_pair():
        local_delta = x.item[0] + x.item[1] + match_shape(next_rdx.item)
        delta = sh2pc_recon(AdditiveSecretSharing(local_delta), party=next_party)
        boolean_share, _ = VSigma.eval(
            delta,
            (next_key.to(DEVICE), next_c.to(DEVICE)),
            next_party.party_id,
        )
        return sh2pc_b2a(boolean_share, next_party)

    if party.party_id in (0, 1):
        v1_a = evaluate_previous_pair()
        v2_a = evaluate_next_pair()
    elif party.party_id == 2:
        v2_a = evaluate_next_pair()
        v1_a = evaluate_previous_pair()
    else:
        raise ValueError("party_id must be 0, 1, or 2")
    if share_table is None:
        share_table, mac_key, smac_table = party.get_param(MACKey, num)
    else:
        share_table, smac_table = share_table
        mac_key = None
    v1 = (share_table.item[0] * v1_a.item)
    v2 = (share_table.item[1] * v2_a.item)

    mac_v1 = RingTensor.mul(smac_table.item[0], v1_a.item)
    mac_v2 = RingTensor.mul(smac_table.item[1], v2_a.item)

    v = v1 + v2
    mac_v = mac_v1 + mac_v2

    v = ReplicatedSecretSharing.reshare(v, party)
    mac_v = ReplicatedSecretSharing.reshare(mac_v, party)

    if mac_key is not None and shape:
        mac_key = mac_key.expand(*shape) if mac_key.numel() == 1 else mac_key.reshape(shape)
    return v.reshape(shape), mac_v.reshape(shape), mac_key


class MACKey(Parameter):
    """
    A class representing the Message Authentication Code (MAC) keys and tables used in the protocol.

    Attributes:
        share_table (ReplicatedSecretSharing): The share of the table (usually ones).
        mac_key (ReplicatedSecretSharing): The share of the MAC key (alpha).
        smac_table (ReplicatedSecretSharing): The share of the MAC of the table (table * alpha).
    """

    def __init__(self, share_table=None, mac_key=None, smac_table=None):
        """
        Initializes the MACKey instance.

        Args:
            share_table (ReplicatedSecretSharing, optional): The share of the table. Defaults to None.
            mac_key (ReplicatedSecretSharing, optional): The share of the MAC key. Defaults to None.
            smac_table (ReplicatedSecretSharing, optional): The share of the MAC of the table. Defaults to None.
        """
        self.share_table = share_table
        self.mac_key = mac_key
        self.smac_table = smac_table

    def __iter__(self):
        return iter([self.share_table, self.mac_key, self.smac_table])

    @staticmethod
    def gen(num_of_keys: int) -> list[Parameter]:
        """
        Generates a list of MACKey parameters for a specified number of keys.

        This method generates random MAC keys and shares them among parties.
        It creates a table of ones, a random MAC key, and computes the MAC of the table.
        Then it shares these values using Replicated Secret Sharing.

        Args:
            num_of_keys: The number of keys to generate.

        Returns:
            list[Parameter]: A list of MACKey instances, one for each party (usually 3 for 3PC).

        Examples:
            >>> keys = MACKey.gen(100)
        """
        table1 = RingTensor.ones([num_of_keys, 1])

        share_tables = ReplicatedSecretSharing.share(table1)

        mac_key = RingTensor.random([num_of_keys, 1])
        share_keys = ReplicatedSecretSharing.share(mac_key)

        mac_table = table1 * mac_key
        smac_tables = ReplicatedSecretSharing.share(mac_table)

        return [MACKey(a, b, c) for a, b, c in zip(share_tables, share_keys, smac_tables)]
