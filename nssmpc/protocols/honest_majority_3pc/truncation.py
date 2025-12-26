#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
import os

from nssmpc.config import SCALE, param_path, HALF_RING
from nssmpc.infra.mpc.aux_parameter.parameter import Parameter
from nssmpc.infra.mpc.party import Party, PartyCtx
from nssmpc.infra.tensor import RingTensor
from nssmpc.primitives.secret_sharing import ReplicatedSecretSharing
from nssmpc.protocols.honest_majority_3pc.base import hm3pc_open


def hm3pc_truncate_aby3(share: ReplicatedSecretSharing, scale: int = SCALE,
                        party: Party = None) -> ReplicatedSecretSharing:
    """Truncate the Replicated Secret Sharing (RSS) input using the ABY3 protocol.

    This method reduces the scale of the fixed-point number by shifting, while handling the
    potential wrap-around (overflow) issues inherent in ring arithmetic.

    Args:
        share: The RSS share to be truncated.
        scale: The number of bits to shift (truncate). Defaults to global SCALE.
        party: The party instance managing the communication. Defaults to None.

    Returns:
        The truncated result with restored scale.

    Examples:
        >>> res = hm3pc_truncate_aby3(share)
    """
    if scale == 1:
        return share
    if party is None:
        party = PartyCtx.get()
    # tag = 'RssTruncAuxParams' if scale == SCALE else f'RssTruncAuxParams_{scale}'
    r, r_t = party.get_param(RssTruncAuxParams, share.numel())
    shape = share.shape
    share = share.flatten()
    r_t.dtype = 'float'
    r.dtype = 'float'
    delta_share = share - r
    delta = hm3pc_open(delta_share)
    delta_trunc = delta // scale
    result = r_t + delta_trunc
    return result.reshape(shape)


class RssTruncAuxParams(Parameter):
    """A class used for generating relevant parameters for truncation of RSS and save the parameters.

    Attributes:
        r (RingTensor): The auxiliary parameter for truncation of RSS.
        r_hat (RingTensor): The auxiliary parameter for truncation of RSS.
        size (int): Size of the parameters.
    """

    def __init__(self):
        """Initializes the RssTruncAuxParams instance.

        Examples:
            >>> param = RssTruncAuxParams()
        """
        self.r = None
        self.r_hat = None
        self.size = 0

    def __iter__(self):
        """Make an instance of this class iterable.

        Returns:
            tuple: A tuple contains the attributes r and r_hat.

        Examples:
            >>> r, r_hat = iter(param)
        """
        return iter((self.r, self.r_hat))

    @staticmethod
    def gen(num_of_params: int, scale: int = SCALE):
        """Generates parameters for truncation operations of RSS.

        Args:
            num_of_params: The number of params to generate.
            scale: The scale of the number to be truncated, defaults to SCALE.

        Returns:
            List[RssTruncAuxParams]: The generated parameters for three parties.

        Examples:
            >>> params = RssTruncAuxParams.gen(100)
        """
        r_hat = RingTensor.random([num_of_params], down_bound=-HALF_RING // (2 * scale),
                                  upper_bound=HALF_RING // (2 * scale))
        r = r_hat * scale
        r_list = ReplicatedSecretSharing.share(r)
        r_hat_list = ReplicatedSecretSharing.share(r_hat)
        aux_params = []
        for i in range(3):
            param = RssTruncAuxParams()
            param.r = r_list[i].to('cpu')
            param.r_hat = r_hat_list[i].to('cpu')
            param.size = num_of_params
            aux_params.append(param)
        return aux_params

