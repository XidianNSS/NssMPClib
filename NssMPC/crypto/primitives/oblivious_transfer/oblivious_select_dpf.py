#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
The implementation is based on the `work of Bai J, Song X, Zhang X, et al. Mostree: Malicious Secure Private Decision Tree Evaluation with Sublinear Communication 2023: 799-813`.
For reference, see the `paper <https://dl.acm.org/doi/abs/10.1145/3627106.3627131>`_.
"""
from NssMPC import RingTensor
from NssMPC.common.utils.common_utils import list_rotate
from NssMPC.crypto.aux_parameter.select_keys.vos_key import VOSKey
from NssMPC.crypto.primitives import ReplicatedSecretSharing
from NssMPC.crypto.primitives.function_secret_sharing.vdpf import VDPF
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import recon


class ObliviousSelect(object):
    @staticmethod
    def check_keys_and_r():
        pass

    @staticmethod
    def selection(table: ReplicatedSecretSharing, idx: ReplicatedSecretSharing):
        idx = idx.view(-1, 1)
        table = table.unsqueeze(1)
        num = idx.shape[0]
        party = table.party
        self_rdx1, _ = party.get_param(f'VOSKey_{party.party_id}_0', num)
        self_rdx0, _ = party.get_param(f'VOSKey_{party.party_id}_1', num)
        next_party = party.virtual_party_with_next
        pre_party = party.virtual_party_with_previous
        key_from_next = next_party.get_param(VOSKey, num)
        key_from_previous = pre_party.get_param(VOSKey, num)
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

        dt1_list.append(recon(delta1, 0))
        dt2_list.append(recon(delta2, 0))

        dt1_list.append(recon(delta2, 1))
        dt2_list.append(recon(delta0, 1))

        dt1_list.append(recon(delta0, 2))
        dt2_list.append(recon(delta1, 2))

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
