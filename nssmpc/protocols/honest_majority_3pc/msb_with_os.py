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
from nssmpc.infra.utils.common import list_rotate
from nssmpc.primitives.secret_sharing import ReplicatedSecretSharing
from nssmpc.primitives.secret_sharing.function import VSigma, VSigmaKey
from nssmpc.protocols.honest_majority_3pc.base import hm3pc_recon
from nssmpc.protocols.semi_honest_2pc.b2a import sh2pc_b2a


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

    next_party = party.virtual_party_with_next
    pre_party = party.virtual_party_with_previous
    self_rdx0, _, _ = next_party.get_param(VSigmaKey, num, tag=f'{party.party_id}')
    self_rdx1, _, _ = pre_party.get_param(VSigmaKey, num, tag=f'{party.party_id}')
    rdx0, k0, c0 = next_party.get_param(VSigmaKey, num, tag=f'{(party.party_id + 1) % 3}')
    rdx1, k1, c1 = pre_party.get_param(VSigmaKey, num, tag=f'{(party.party_id - 1) % 3}')

    rdx_list = [ReplicatedSecretSharing([self_rdx0.item, self_rdx1.item]),
                ReplicatedSecretSharing([RingTensor.convert_to_ring(0), rdx1.item]),
                ReplicatedSecretSharing([rdx0.item, RingTensor.convert_to_ring(0)])]

    rdx_list = list_rotate(rdx_list, party.party_id)

    delta0 = x + rdx_list[0]
    delta1 = x + rdx_list[1]
    delta2 = x + rdx_list[2]

    dt1_list = []
    dt2_list = []

    dt1_list.append(hm3pc_recon(delta1, 0))
    dt2_list.append(hm3pc_recon(delta2, 0))

    dt1_list.append(hm3pc_recon(delta2, 1))
    dt2_list.append(hm3pc_recon(delta0, 1))

    dt1_list.append(hm3pc_recon(delta0, 2))
    dt2_list.append(hm3pc_recon(delta1, 2))

    dt1 = dt1_list[party.party_id]
    dt2 = dt2_list[party.party_id]

    v1, pi1 = VSigma.eval(dt1, (k1.to(DEVICE), c1.to(DEVICE)), 0)
    v2, pi2 = VSigma.eval(dt2, (k0.to(DEVICE), c0.to(DEVICE)), 1)

    if party.party_id == 0:
        v1_a = sh2pc_b2a(v1, party.virtual_party_with_previous)
        v2_a = sh2pc_b2a(v2, party.virtual_party_with_next)
    elif party.party_id == 1:
        v1_a = sh2pc_b2a(v1, party.virtual_party_with_previous)
        v2_a = sh2pc_b2a(v2, party.virtual_party_with_next)
    elif party.party_id == 2:
        v2_a = sh2pc_b2a(v2, party.virtual_party_with_next)
        v1_a = sh2pc_b2a(v1, party.virtual_party_with_previous)
    else:
        raise Exception("Party id error!")
    if share_table is None:
        share_table, mac_key, smac_table = party.get_param(MACKey, num)
    else:
        share_table = share_table[0]
        smac_table = share_table[1]
        mac_key = None
    v1 = (share_table.item[0] * v1_a.item)
    v2 = (share_table.item[1] * v2_a.item)

    mac_v1 = RingTensor.mul(smac_table.item[0], v1_a.item)
    mac_v2 = RingTensor.mul(smac_table.item[1], v2_a.item)

    v = v1 + v2
    mac_v = mac_v1 + mac_v2

    v = ReplicatedSecretSharing.reshare(v, party)
    mac_v = ReplicatedSecretSharing.reshare(mac_v, party)

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
