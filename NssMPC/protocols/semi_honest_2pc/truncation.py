#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
from typing import List, Tuple

import torch

from NssMPC import RingTensor
from NssMPC.config import data_type, RING_MAX, DEBUG_LEVEL
from NssMPC.infra.mpc.param_provider.parameter import Parameter
from NssMPC.infra.mpc.party import Party, PartyCtx
from NssMPC.primitives.secret_sharing import AdditiveSecretSharing


def truncate(share: AdditiveSecretSharing, scale: int = None,
             party: Party = None) -> AdditiveSecretSharing:
    """
    Truncate the Additive Secret Sharing (ASS) input with a given scale.

    This method implements the truncation protocol from `CrypTen <https://proceedings.neurips.cc/paper_files/paper/2021/file/2754518221cfbc8d25c13a06a4cb8421-Paper.pdf>`_.
    It addresses the "probabilistic truncation" issue where simply shifting local shares fails
    if the sum of shares wraps around the ring size 2^{64}.

    Args:
        share: The sharing data to be truncated.
        scale: The amount of scaling factor to remove (number of bits). Defaults to None.
        party: The party instance managing the communication. Defaults to None.

    Returns:
        The truncated sharing data.

    Examples:
        >>> res = truncate(share, scale=16)
    """
    if party is None:
        party = PartyCtx.get()
    share_tensor = share.item.tensor
    wrap_count = _wraps(share_tensor, party)
    share_tensor = share_tensor.div(scale, rounding_mode="trunc").to(data_type)
    share_tensor -= wrap_count * 4 * (RING_MAX // 4 // scale)
    return AdditiveSecretSharing(RingTensor(share_tensor, share.dtype, share.device))


def _wraps(share_tensor: torch.Tensor, party: Party) -> torch.Tensor:
    """
    Privately computes the number of wraparounds for a set of shares.

    This method supports computation of the wrap count for a set of arithmetically shared tensor,
    which is based on the corresponding method in `CrypTen <https://proceedings.neurips.cc/paper_files/paper/2021/file/2754518221cfbc8d25c13a06a4cb8421-Supplemental.pdf>`_.

    According to the paper, we can note that:
        [theta_x] = theta_z + [beta_xr] - [theta_r] - [eta_xr]

    Where [theta_i] is the wraps for a variable i
          [beta_ij] is the differential wraps for variables i and j
          [eta_ij]  is the plaintext wraps for variables i and j

    Note:
        Since [eta_xr] = 0 with probability 1 - |x| / Q for modulus Q, we
        can make the assumption that [eta_xr] = 0 with high probability.

    Args:
        share_tensor: The shared tensor to compute the number of wraparounds.
        party: The party instance.

    Returns:
        The number of wraps of the shares.
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
        z_other = party.recv()
        theta_z = Wrap.count_wraps([z_other, z])

        return theta_z + beta_xr - theta_r


class Wrap(Parameter):
    """
    A class used for operations related to the truncation, including methods for generating relevant parameters and calculating the number of wraps.
    """

    def __init__(self, r: RingTensor = None, theta_r: RingTensor = None):
        """
        Initializes the Wrap parameter.

        Attributes:
            r (ArithmeticSecretSharing): Generated random numbers, used to obfuscate real input data in the secure truncation protocol.
            theta_r (ArithmeticSecretSharing): Parameters used in truncation.
        """
        self.r = r
        self.theta_r = theta_r

    @staticmethod
    def gen(num_of_params: int) -> Tuple['Wrap', 'Wrap']:
        """
        Generates parameters for wrap-related operations.

        Args:
            num_of_params: The number of params to generate.

        Returns:
            The generated parameters for two parties.

        Examples:
            >>> w0, w1 = Wrap.gen(10)
        """
        r = RingTensor.random([num_of_params])
        r0, r1 = AdditiveSecretSharing.share(r, 2)
        theta_r = Wrap.count_wraps([r0.item.tensor, r1.item.tensor])

        theta_r0, theta_r1 = AdditiveSecretSharing.share(RingTensor(theta_r), 2)

        wrap_0 = Wrap(r0.item.tensor, theta_r0.item.tensor)
        wrap_1 = Wrap(r1.item.tensor, theta_r1.item.tensor)

        return wrap_0, wrap_1

    @staticmethod
    def count_wraps(share_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes the number of overflows or underflows in a set of shares.

        We compute this by counting the number of overflows and underflows as we
        traverse the list of shares.

        Args:
            share_list: The list contains the shares to compute warps.

        Returns:
            The number of overflows or underflows, with overflows being positive and underflows being negative.

        Examples:
            >>> wraps = Wrap.count_wraps([share1, share2])
        """
        result = torch.zeros_like(share_list[0], dtype=data_type)
        prev = share_list[0]
        for cur in share_list[1:]:
            next = cur + prev
            result -= ((prev < 0) & (cur < 0) & (next > 0)).to(data_type)  # underflow
            result += ((prev > 0) & (cur > 0) & (next < 0)).to(data_type)  # overflow
            prev = next
        return result
