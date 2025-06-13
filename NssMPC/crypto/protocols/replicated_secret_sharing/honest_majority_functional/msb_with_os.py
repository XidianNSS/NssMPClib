#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
The implementation is based on the work of `Fu J, Cheng K, Xia Y, et al. Private decision tree evaluation with malicious security via function secret sharing 2024: 310-330`.
For reference, see the `paper <https://link.springer.com/chapter/10.1007/978-3-031-70890-9_16>`_.
"""
from NssMPC import RingTensor
from NssMPC.common.utils.common_utils import list_rotate
from NssMPC.config import DEVICE
from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.aux_parameter import MACKey
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.vsigma_key import VSigmaKey
from NssMPC.crypto.primitives.function_secret_sharing.vsigma import VSigma

from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional import b2a
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import recon


def msb_with_os_without_mac_check(x, share_table=None):
    shape = x.shape
    x = x.reshape(-1, 1)
    num = x.shape[0]
    party = PartyRuntime.party

    self_rdx0, _, _ = party.get_param(f'VSigmaKey_{party.party_id}_0', num)
    self_rdx1, _, _ = party.get_param(f'VSigmaKey_{party.party_id}_1', num)
    next_party = party.virtual_party_with_next
    pre_party = party.virtual_party_with_previous
    key_from_next = next_party.get_param(VSigmaKey, num)
    key_from_previous = pre_party.get_param(VSigmaKey, num)
    rdx1, k1, c1 = key_from_previous
    rdx0, k0, c0 = key_from_next

    rdx_list = [x.__class__([self_rdx0.item, self_rdx1.item]),
                x.__class__([RingTensor.convert_to_ring(0), rdx1.item]),
                x.__class__([rdx0.item, RingTensor.convert_to_ring(0)])
                ]

    rdx_list = list_rotate(rdx_list, party.party_id)

    delta0 = x + rdx_list[0]
    delta1 = x + rdx_list[1]
    delta2 = x + rdx_list[2]

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

    v1, pi1 = VSigma.eval(dt1, (k1.to(DEVICE), c1.to(DEVICE)), 0)
    v2, pi2 = VSigma.eval(dt2, (k0.to(DEVICE), c0.to(DEVICE)), 1)

    if party.party_id == 0:
        v1_a = b2a(v1, party.virtual_party_with_previous)
        v2_a = b2a(v2, party.virtual_party_with_next)
    elif party.party_id == 1:
        v1_a = b2a(v1, party.virtual_party_with_previous)
        v2_a = b2a(v2, party.virtual_party_with_next)
    elif party.party_id == 2:
        v2_a = b2a(v2, party.virtual_party_with_next)
        v1_a = b2a(v1, party.virtual_party_with_previous)
    else:
        raise Exception("Party id error!")
    if share_table is None:
        share_table, mac_key, smac_table = PartyRuntime.party.get_param(MACKey, num)
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

    v = x.__class__.reshare(v, party)
    mac_v = x.__class__.reshare(mac_v, party)

    return v.reshape(shape), mac_v.reshape(shape), mac_key
