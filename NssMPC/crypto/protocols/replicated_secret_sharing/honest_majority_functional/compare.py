#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC import RingTensor
from NssMPC.config.runtime import MAC_BUFFER
from NssMPC.crypto.aux_parameter import Parameter
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import coin, check_zero, open
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.msb_with_os import \
    msb_with_os_without_mac_check
from NssMPC.crypto.protocols.replicated_secret_sharing.semi_honest_functional import rand_like, mul_with_out_trunc


def secure_ge(x, y):  # TODO support float
    """
    Gets the most significant bit of an RSS sharing ⟨x⟩
    :param x: an RSS sharing ⟨x⟩
    :param y: an RSS sharing ⟨y⟩
    :return: the most significant bit of x
    """
    z = x - y
    dtype = x.dtype
    x.dtype = y.dtype = 'int'

    shape = z.shape

    res, mac_v, mac_key = msb_with_os_without_mac_check(z)

    MAC_BUFFER.add(res, mac_v, mac_key)
    # mac_check(res, mac_v, mac_key)

    res.dtype = dtype
    return (res.view(shape) * res.scale - 1) * -1


def mac_check(x, mx, mac_key):
    r = rand_like(x, x.party)
    mr = mul_with_out_trunc(r, mac_key)
    ro = coin(x.numel(), x.party).reshape(x.shape)
    v = r + x * ro
    w = mr + mx * ro
    v = open(v)
    check_zero(w - mac_key * v)


class MACKey(Parameter):
    def __init__(self, share_table=None, mac_key=None, smac_table=None):
        self.share_table = share_table
        self.mac_key = mac_key
        self.smac_table = smac_table

    def __iter__(self):
        return iter([self.share_table, self.mac_key, self.smac_table])

    @staticmethod
    def gen(num_of_keys):
        table_01 = RingTensor.zeros([num_of_keys, 2])
        table_01[:, 1] += 1

        from NssMPC import ReplicatedSecretSharing
        share_tables = ReplicatedSecretSharing.share(table_01)

        mac_key = RingTensor.random([num_of_keys, 1])
        share_keys = ReplicatedSecretSharing.share(mac_key)

        mac_table = table_01 * mac_key
        smac_tables = ReplicatedSecretSharing.share(mac_table)

        return [MACKey(a, b, c) for a, b, c in zip(share_tables, share_keys, smac_tables)]
