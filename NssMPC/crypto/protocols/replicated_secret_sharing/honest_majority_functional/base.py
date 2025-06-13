#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.common.ring import RingTensor
from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.protocols.replicated_secret_sharing.semi_honest_functional.random import rand, rand_like
from NssMPC.crypto.protocols.replicated_secret_sharing.semi_honest_functional.multiplication import mul_with_out_trunc
from NssMPC.config.configs import DTYPE


def open(x):
    """
    Open an RSS sharing <x> to each party, and each party will get the restored `x`.

    :param x: An RSS sharing <x>
    :type x: ReplicatedSecretSharing
    :return: The restore value of <x>
    :rtype: RingTensor
    :raise ValueError: If the share received from the other two parties are not equal.
    """
    party = PartyRuntime.party
    # send x0 to P_{i+1}
    party.send((party.party_id + 1) % 3, x.item[0])
    # receive x2 from P_{i+1}
    x_2_0 = party.receive((party.party_id + 2) % 3)
    # send x1 to P_{i-1}
    party.send((party.party_id + 2) % 3, x.item[1])
    # receive x2 from P_{i-1}
    x_2_1 = party.receive((party.party_id + 1) % 3)

    cmp = x_2_0 - x_2_1
    # print(cmp)
    cmp = cmp.tensor.flatten().sum(axis=0)
    if cmp != 0:
        raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")

    return x.item[0] + x.item[1] + x_2_0


def coin(num_of_value, party):
    """
    Outputs a random value r to all the parties.

    :param num_of_value: The number of random value to generate.
    :type num_of_value: int or list
    :param party: Party that hold the random value.
    :type party: Party
    :return: The random value r.
    :rtype: ReplicatedSecretSharing
    """
    rss_r = rand([num_of_value], party)
    r = open(rss_r)
    return r


def recon(x, party_id):
    """
    Reconstructs a consistent RSS sharing ⟨x⟩ to P_i.

    :param x: The RSS sharing ⟨x⟩ we use to reconstruct the secret to P_i.
    :type x: ReplicatedSecretSharing
    :param party_id: the party id of P_i
    :type party_id: int
    :return: plain text x only known to P_i
    :rtype: RingTensor
    """
    party = PartyRuntime.party
    # P_{i+1} send x_{i+2} to P_{i}
    if party.party_id == (party_id + 1) % 3:
        party.send(party_id, x.item[1])
    # P_{i-1} send x_{i+2} to P_{i}
    elif party.party_id == (party_id + 2) % 3:
        party.send(party_id, x.item[0])

    elif party.party_id == party_id:
        x_2_0 = party.receive((party.party_id + 2) % 3)
        x_2_1 = party.receive((party.party_id + 1) % 3)

        cmp = x_2_0 - x_2_1
        cmp = cmp.tensor.flatten().sum(axis=0)
        if cmp != 0:
            raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")
        if party.party_id == party_id:
            return x.item[0] + x.item[1] + x_2_0
    return None


def receive_share_from(input_id, party):
    """
    Receives a shared value from P_i.

    :param party: The party that receive the share.
    :type party: Party
    :param input_id: The id of input party.
    :type input_id: int
    :return: An RSS sharing ⟨x⟩
    :rtype: ReplicatedSecretSharing
    """
    # receive shape from P_i
    shape_of_received_shares = party.receive(input_id)
    r = rand(shape_of_received_shares, party)
    _ = recon(r, input_id)

    # receive delta from P_{input_id}
    delta = party.receive(input_id)

    # check if delta is same
    other_id = (0 + 1 + 2) - party.party_id - input_id
    # send delta to P_{other_id}
    party.send(other_id, delta)
    delta_other = party.receive(other_id)

    cmp = delta - delta_other
    cmp = cmp.tensor.flatten().sum(axis=0)
    if cmp != 0:
        raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")
    res = r + delta
    res.dtype = delta.dtype
    return res


def share(x, party):
    """
    Shares a secret x from P_i among three parties, and need the other two parties to receive.

    :param x: The public input x only input by P_i.
    :type x: RingTensor
    :param party: The party from which we share the secret.
    :type party: Party
    :return: An RSS sharing ⟨x⟩.
    :rtype: ReplicatedSecretSharing
    """
    if not isinstance(x, RingTensor):
        raise TypeError("unsupported data type(s) ")

    # 这里share的应该是个RingTensor
    shape = x.shape
    # send shape to P_{i+1} and P_{i-1}
    party.send((party.party_id + 1) % 3, shape)
    party.send((party.party_id + 2) % 3, shape)
    r = rand_like(x, party)
    r_recon = recon(r, party.party_id)
    delta = x - r_recon
    # broadcasts delta to all parties
    party.send((party.party_id + 1) % 3, delta)
    party.send((party.party_id + 2) % 3, delta)
    res = r + delta
    return res


def mac_check(x, mx, mac_key):
    """
    Verifies the message authentication code (MAC) of the input `x` to check if it has been tampered with.

    :param x: The message to be verified.
    :type x: ReplicatedSecretSharing
    :param mx: The MAC value of the message.
    :type mx: ReplicatedSecretSharing
    :param mac_key: The key used for verification.
    :type mac_key: MACKey
    :return: The verification result. Returns 1 if successful, otherwise 0.
    :rtype: int
    """
    party = PartyRuntime.party
    r = rand_like(x, party)
    mr = mul_with_out_trunc(r, mac_key)
    ro = coin(x.numel(), party).reshape(x.shape)
    v = r + x * ro
    w = mr + mx * ro
    v = open(v)
    check_zero(w - mac_key * v)


def check_zero(x):
    """
    Checks if an RSS sharing ⟨x⟩ is zero.

    :param x: An RSS sharing ⟨x⟩.
    :type x: ReplicatedSecretSharing
    :return: 1 if x is zero, 0 otherwise.
    :rtype: int
    """
    # print("x restore", x.restore())
    r = rand_like(x, PartyRuntime.party)
    w = mul_with_out_trunc(x, r)
    w_open = open(w)
    res = (w_open.tensor == 0) + 0
    if res.flatten().sum() != x.numel():
        raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")
    return res


def check_is_all_element_equal(x, y):
    """
    Checks if all elements of x are equal to all elements of y.

    :param x: A RingTensor x.
    :type x: RingTensor
    :param y: A RingTensor y.
    :type y: RingTensor
    :return: 1 if all elements of x are equal to all elements of y, 0 otherwise
    :rtype: int
    """
    cmp = x - y
    cmp = cmp.tensor.flatten().sum(axis=0)
    if cmp != 0:
        raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")
    return 1
