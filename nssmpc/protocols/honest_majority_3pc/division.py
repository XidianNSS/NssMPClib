#  This file is part of the NssMPClib project.
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from nssmpc.primitives.secret_sharing import ReplicatedSecretSharing
from nssmpc.protocols.honest_majority_3pc.mac_check import MAC_BUFFER
from nssmpc.protocols.semi_honest_3pc.division import sh3pc_div


def hm3pc_div(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing) -> ReplicatedSecretSharing:
    """Computes fixed-point division and checks the comparison MACs.

    Uses the domain and precision constraints documented by sh3pc_div. The
    honest-majority runtime must be active so all multiplications are checked.

    Args:
        x: The fixed-point replicated dividend.
        y: The nonzero fixed-point replicated divisor, broadcastable with x.

    Returns:
        A replicated sharing of the approximate quotient.

    Raises:
        TypeError: If either input is not a fixed-point sharing.
        RuntimeError: If the input shapes cannot be broadcast together.
        ValueError: If a protocol consistency check fails.

    Examples:
        >>> quotient = hm3pc_div(x, y)
    """
    result = sh3pc_div(x, y)
    MAC_BUFFER.check()
    return result
