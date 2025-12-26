#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
The implementation is based on the `work of Bai J, Song X, Zhang X, et al. Mostree: Malicious Secure Private Decision Tree Evaluation with Sublinear Communication 2023: 799-813`.
For reference, see the `paper <https://dl.acm.org/doi/abs/10.1145/3627106.3627131>`_.
"""
from nssmpc.infra.mpc.aux_parameter import ParamProvider
from nssmpc.infra.mpc.aux_parameter.parameter import Parameter
from nssmpc.infra.mpc.party import PartyCtx, Party3PC
from nssmpc.infra.tensor import RingTensor
from nssmpc.infra.utils.common import list_rotate
from nssmpc.primitives.secret_sharing import ReplicatedSecretSharing
from nssmpc.primitives.secret_sharing.function import VDPF, VDPFKey
from nssmpc.protocols.honest_majority_3pc.base import hm3pc_recon


class ObliviousSelect(object):
    @staticmethod
    def check_keys_and_r():
        pass

    @staticmethod
    def selection(table: ReplicatedSecretSharing, idx: ReplicatedSecretSharing,
                  party: Party3PC = None) -> ReplicatedSecretSharing:
        """
        Performs oblivious selection from a table using the index.

        Args:
            table: The table to select from.
            idx: The index for selection.
            party: The party instance. Defaults to None.

        Returns:
            ReplicatedSecretSharing: The selected value.

        Examples:
            >>> res = ObliviousSelect.selection(table, idx)
        """
        if party is None:
            party = PartyCtx.get()
        idx = idx.view(-1, 1)
        table = table.unsqueeze(1)
        num = idx.shape[0]
        next_party = party.virtual_party_with_next
        pre_party = party.virtual_party_with_previous
        self_rdx1, _ = pre_party.get_param(VOSKey, num, tag=f'{party.party_id}')
        self_rdx0, _ = next_party.get_param(VOSKey, num, tag=f'{party.party_id}')
        key_from_next = next_party.get_param(VOSKey, num, tag=f'{(party.party_id + 1) % 3}')
        key_from_previous = pre_party.get_param(VOSKey, num, tag=f'{(party.party_id - 1) % 3}')
        rdx1, k1 = key_from_previous
        rdx0, k0 = key_from_next

        rdx_list = [ReplicatedSecretSharing([self_rdx1, self_rdx0]).reshape(-1, 1),
                    ReplicatedSecretSharing([RingTensor.convert_to_ring(0), rdx1], ).reshape(-1, 1),
                    ReplicatedSecretSharing([rdx0, RingTensor.convert_to_ring(0)]).reshape(-1, 1)
                    ]

        rdx_list = list_rotate(rdx_list, party.party_id)

        delta0 = idx - rdx_list[0]
        delta1 = idx - rdx_list[1]
        delta2 = idx - rdx_list[2]

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

        j = RingTensor.arange(start=0, end=table.shape[-1], device=idx.device)

        v1, pi1 = VDPF.one_key_eval((j - dt1), k1, table.shape[-1], 0)
        v2, pi2 = VDPF.one_key_eval((j - dt2), k0, table.shape[-1], 1)

        v1 = RingTensor(v1, idx.dtype, idx.device)
        v2 = RingTensor(v2, idx.dtype, idx.device)

        res1 = (table.item[0] * v1).sum(-1)
        res2 = (table.item[1] * v2).sum(-1)

        res = res1 + res2
        return ReplicatedSecretSharing.reshare(res, party)


class VOSKey(Parameter):
    """
    Parameter class for VOS keys.
    """

    def __init__(self, r00=None, k00=VDPFKey()):
        """
        Initializes VOSKey.

        Args:
            r00 (RingTensor, optional): Random value r00. Defaults to None.
            k00 (VDPFKey, optional): VDPF key. Defaults to VDPFKey().
        """
        self.r00 = r00
        self.k00 = k00

    def __iter__(self):
        return iter([self.r00, self.k00])

    @staticmethod
    def gen(num_of_keys: int):
        """
        Generates VOS keys.

        Args:
            num_of_keys: Number of keys to generate.

        Returns:
            tuple: A tuple of two VOSKey instances.

        Examples:
            >>> key1, key2 = VOSKey.gen(10)
        """
        r00 = RingTensor.random([num_of_keys])
        r01 = RingTensor.random([num_of_keys])
        r0 = r00 + r01
        k00, k01 = VDPFKey.gen(num_of_keys, r0, RingTensor.convert_to_ring(1))
        VOSKey.check_keys_and_r()
        return VOSKey(r00=r00, k00=k00), VOSKey(r00=r01, k00=k01)

    @staticmethod
    def check_keys_and_r():
        pass
