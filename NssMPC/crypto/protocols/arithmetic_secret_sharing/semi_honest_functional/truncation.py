#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC import RingTensor
from NssMPC.config import data_type, RING_MAX, DEBUG_LEVEL
from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.aux_parameter import Wrap


def truncate(share, scale=None):
    """
    Truncate the sharing data with a given scale.

    This method supports truncation operation over an ASS instance, which is based on the truncation method in `CrypTen <https://proceedings.neurips.cc/paper_files/paper/2021/file/2754518221cfbc8d25c13a06a4cb8421-Paper.pdf>`_.
    This method solves the problem of inaccurate results caused by truncate directly if the sum of shares `wraps around` the ring size.

    :param share: The sharing data to be truncated.
    :type share: ArithmeticSecretSharing
    :param scale: The scale to be truncated.
    :type scale: int
    :returns: The truncated sharing data.
    :rtype: ArithmeticSecretSharing
    """
    share_tensor = share.item.tensor
    wrap_count = _wraps(share_tensor)
    share_tensor = share_tensor.div(scale, rounding_mode="trunc").to(data_type)
    share_tensor -= wrap_count * 4 * (RING_MAX // 4 // scale)
    return share.__class__(RingTensor(share_tensor, share.dtype, share.device))


def _wraps(share_tensor):
    """
    Privately computes the number of wraparounds for a set of shares.

    This method supports computation of the wrap count for a set of arithmetically shared tensor,
    which is based on the corresponding method in `CrypTen <https://proceedings.neurips.cc/paper_files/paper/2021/file/2754518221cfbc8d25c13a06a4cb8421-Supplemental.pdf>`_.

    According to the paper, we can note that:
        [theta_x] = theta_z + [beta_xr] - [theta_r] - [eta_xr]

    Where [theta_i] is the wraps for a variable i
          [beta_ij] is the differential wraps for variables i and j
          [eta_ij]  is the plaintext wraps for variables i and j

    .. note::
        Since [eta_xr] = 0 with probability 1 - `|x|` / Q for modulus Q, we
        can make the assumption that [eta_xr] = 0 with high probability.

    :param share_tensor: The shared tensor to compute the number of wraparounds.
    :type share_tensor: torch.Tensor
    :returns: The number of wraps of the shares.
    :rtype: torch.Tensor

    """
    party = PartyRuntime.party
    wrap = party.get_param(Wrap, share_tensor.numel())
    r, theta_r = wrap.r, wrap.theta_r
    if not (DEBUG_LEVEL == 2):
        r = r.reshape(share_tensor.shape)
        theta_r = theta_r.reshape(share_tensor.shape)

    beta_xr = Wrap.count_wraps([share_tensor, r])
    z = share_tensor + r

    if party.party_id == 0:
        party.send(z)
        return beta_xr - theta_r

    if party.party_id == 1:
        z_other = party.receive()
        theta_z = Wrap.count_wraps([z_other, z])

        return theta_z + beta_xr - theta_r
