from NssMPC.infra.tensor import RingTensor
from NssMPC.infra.mpc.party import Party, PartyCtx
from NssMPC.primitives.secret_sharing import ReplicatedSecretSharing
from NssMPC.protocols.honest_majority_3pc.base import open, coin
from NssMPC.protocols.semi_honest_3pc.multiplication import mul_with_out_trunc
from NssMPC.protocols.semi_honest_3pc.random import rand_like


class MacBuffer:
    """Manage the operation of secret shared values and validate them with message verification codes.

    This class provides the buffer to add, store, check and other functions.

    Attributes:
        x (list): Data storage.
        mac (list): Store the corresponding message authentication code (MAC).
        key (list): Store the corresponding key.
    """

    def __init__(self):
        """Initializes the MacBuffer.

        Examples:
            >>> buffer = MacBuffer()
        """
        self.x = []
        self.mac = []
        self.key = []

    def add(self, x, mac, key):
        """Adds new data, MAC, and keys to the MacBuffer.

        Args:
            x (ReplicatedSecretSharing): Data item.
            mac (ReplicatedSecretSharing): Message authentication code.
            key (ReplicatedSecretSharing): Secret key.

        Examples:
            >>> buffer.add(x, mac, key)
        """
        self.x.append(x.clone())
        self.mac.append(mac.clone())
        self.key.append(key.clone())

    def check(self):
        """Checks whether the combined data and MAC are consistent.

        First, merge three lists, and then verify that the combined data and MAC are consistent.
        After the validation is complete, resets the instance.

        Examples:
            >>> buffer.check()
        """
        x = ReplicatedSecretSharing.cat(self.x)
        mac = ReplicatedSecretSharing.cat(self.mac)
        key = ReplicatedSecretSharing.cat(self.key)
        mac_check(x, mac, key)
        self.__init__()


MAC_BUFFER = MacBuffer()


def check_zero(x: ReplicatedSecretSharing, party: Party | None = None) -> int:
    """
    Verifies if the reconstructed value of a secret sharing is zero.

    This function performs a secure check to determine if the underlying secret is zero
    without revealing the secret itself (except that it is zero). It uses a random
    masking technique.

    Args:
        x: The secret shares to check.
        party: The party instance. If None, uses the current party context.

    Returns:
        int: 1 if the value is zero, 0 otherwise.

    Raises:
        ValueError: If the consistency check fails (potential malicious party).

    Examples:
        >>> is_zero = check_zero(x_share)
    """
    if party is None:
        party = PartyCtx.get()
    # print("x restore", x.restore())
    r = rand_like(x, party)
    w = mul_with_out_trunc(x, r)
    w_open = open(w)
    res = (w_open.tensor == 0) + 0
    if res.flatten().sum() != x.numel():
        raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")
    return res


def check_is_all_element_equal(x: RingTensor, y: RingTensor) -> int:
    """
    Checks if two RingTensors are identical across parties.

    This function compares two tensors and verifies that they are equal. It is typically
    used for consistency checks between parties.

    Args:
        x: The first tensor.
        y: The second tensor.

    Returns:
        int: 1 if the tensors are equal.

    Raises:
        ValueError: If the tensors are not equal or calculations do not agree.

    Examples:
        >>> check_is_all_element_equal(tensor_a, tensor_b)
    """
    cmp = x - y
    cmp = cmp.tensor.flatten().sum(axis=0)
    if cmp != 0:
        raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")
    return 1


def mac_check(x: ReplicatedSecretSharing, mx: ReplicatedSecretSharing, mac_key: ReplicatedSecretSharing,
              party: Party = None) -> None:
    """
    Verifies the Message Authentication Code (MAC) of a value.

    This function checks if the relationship :math:`m_x = x \cdot \\alpha` holds,
    where `x` is the value, `mx` is the MAC share, and `mac_key` is the global key share.
    It ensures the integrity of the computation.

    Args:
        x: The value to verify.
        mx: The MAC share corresponding to `x`.
        mac_key: The MAC key share (alpha).
        party: The party instance. If None, uses the current party context.

    Raises:
        ValueError: If the MAC verification fails.

    Examples:
        >>> mac_check(x_share, mac_share, key_share)
    """
    if party is None:
        party = PartyCtx.get()
    r = rand_like(x, party)
    mr = mul_with_out_trunc(r, mac_key)
    ro = coin(x.numel(), party).reshape(x.shape)
    v = r + x * ro
    w = mr + mx * ro
    v = open(v)
    check_zero(w - mac_key * v)
