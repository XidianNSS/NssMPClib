from NssMPC import RingTensor
from NssMPC.config import data_type, RING_MAX, DEBUG_LEVEL
from NssMPC.crypto.aux_parameter import Wrap


def truncate(share, scale=None):
    """
    CrypTen-based truncation methods

    Args:
        share (ArithmeticSharedRingTensor): The sharing data to be truncated
        scale: truncation bits

    Returns:
        ArithmeticShardRingTensor: The truncated sharing data
    """
    share_tensor = share.item.tensor
    wrap_count = _wraps(share_tensor, share.party)
    share_tensor = share_tensor.div(scale, rounding_mode="trunc").to(data_type)
    share_tensor -= wrap_count * (((RING_MAX // 4) // scale) * 4 - 1)
    return share.__class__(RingTensor(share_tensor, share.dtype, share.device), share.party)


def _wraps(share_tensor, party):
    """
    Privately computes the number of wraparounds for a set a shares
    To do so, we note that:
        [theta_x] = theta_z + [beta_xr] - [theta_r] - [eta_xr]

    Where [theta_i] is the wraps for a variable i
          [beta_ij] is the differential wraps for variables i and j
          [eta_ij]  is the plaintext wraps for variables i and j

    Note: Since [eta_xr] = 0 with probability 1 - |x| / Q for modulus Q, we
    can make the assumption that [eta_xr] = 0 with high probability.

    Based on the situation of two parties
    """

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
