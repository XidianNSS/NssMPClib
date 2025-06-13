#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.common.ring import *
from NssMPC.config.configs import SCALE
from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.aux_parameter.truncation_keys.rss_trunc_aux_param import RssTruncAuxParams
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.base import open


def truncate(share, scale=SCALE):
    """
    Truncate the input `share` with method from ABY3.

    :param share: The RSS share to be truncated.
    :type share: ReplicatedSecretSharing
    :param scale: The scale of the number to be truncated, defaults to SCALE.
    :type scale: int
    :return: Result of the truncation.
    :rtype: ReplicatedSecretSharing
    """
    if scale == 1:
        return share
    tag = 'RssTruncAuxParams' if scale == SCALE else f'RssTruncAuxParams_{scale}'
    r, r_t = PartyRuntime.party.get_param(tag, share.numel())
    shape = share.shape
    share = share.flatten()
    r_t.dtype = 'float'
    r.dtype = 'float'
    delta_share = share - r
    delta = open(delta_share)
    delta_trunc = delta // scale
    result = r_t + delta_trunc
    return result.reshape(shape)
