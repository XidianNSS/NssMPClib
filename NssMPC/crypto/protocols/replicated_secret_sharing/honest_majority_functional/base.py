from NssMPC.common.ring import RingTensor
from NssMPC.crypto.protocols.replicated_secret_sharing.semi_honest_functional.random import rand, rand_like
from NssMPC.crypto.protocols.replicated_secret_sharing.semi_honest_functional.multiplication import mul_with_out_trunc
from NssMPC.config.configs import DTYPE


def open(x):
    """
    Open an RSS sharing <x> to each party

    :param x: an RSS sharing <x>
    :return: the restore value of <x>
    """
    # send x0 to P_{i+1}
    x.party.send((x.party.party_id + 1) % 3, x.item[0])
    # receive x2 from P_{i+1}
    x_2_0 = x.party.receive((x.party.party_id + 2) % 3)
    # send x1 to P_{i-1}
    x.party.send((x.party.party_id + 2) % 3, x.item[1])
    # receive x2 from P_{i-1}
    x_2_1 = x.party.receive((x.party.party_id + 1) % 3)

    cmp = x_2_0 - x_2_1
    # print(cmp)
    cmp = cmp.tensor.flatten().sum(axis=0)
    if cmp != 0:
        raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")

    return x.item[0] + x.item[1] + x_2_0


def coin(num_of_value, party):
    """
    Outputs a random value r to all the parties
    :param num_of_value: the number of random value to generate
    :param party: party instance
    :return: random value r
    """
    rss_r = rand([num_of_value], party)
    r = open(rss_r)
    return r


def recon(x, party_id):
    """
    Reconstructs a consistent RSS sharing ⟨x⟩ to P_i
    :param x: an RSS sharing ⟨x⟩
    :param party_id: the party id of P_i
    :return: plain text x only known to P_i
    """
    # P_{i+1} send x_{i+2} to P_{i}
    if x.party.party_id == (party_id + 1) % 3:
        x.party.send(party_id, x.item[1])
    # P_{i-1} send x_{i+2} to P_{i}
    elif x.party.party_id == (party_id + 2) % 3:
        x.party.send(party_id, x.item[0])

    elif x.party.party_id == party_id:
        x_2_0 = x.party.receive((x.party.party_id + 2) % 3)
        x_2_1 = x.party.receive((x.party.party_id + 1) % 3)

        cmp = x_2_0 - x_2_1
        cmp = cmp.tensor.flatten().sum(axis=0)
        if cmp != 0:
            raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")
        if x.party.party_id == party_id:
            return x.item[0] + x.item[1] + x_2_0
    return None


def receive_share_from(input_id, party):
    """
    Receives a shared value from P_i
    :param party: party instance
    :param input_id: the id of input party
    :return: an RSS sharing ⟨x⟩
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
    # todo:不优雅的方式
    res.dtype = DTYPE
    return res


def share(x, party):
    """
    Shares a secret x from P_i among three parties.
    :param x: public input x only input by P_i
    :param party: party instance
    :return: an RSS sharing ⟨x⟩
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
    r = rand_like(x, x.party)
    mr = mul_with_out_trunc(r, mac_key)
    ro = coin(x.numel(), x.party).reshape(x.shape)
    v = r + x * ro
    w = mr + mx * ro
    v = open(v)
    check_zero(w - mac_key * v)


def check_zero(x):
    """
    Checks if an RSS sharing ⟨x⟩ is zero
    :param x: an RSS sharing ⟨x⟩
    :return: 1 if x is zero, 0 otherwise
    """
    # print("x restore", x.restore())
    r = rand_like(x, x.party)
    w = mul_with_out_trunc(x, r)
    w_open = open(w)
    res = (w_open.tensor == 0) + 0
    if res.flatten().sum() != x.numel():
        raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")
    return res


def check_is_all_element_equal(x, y):
    """
    Checks if all elements of x are equal to all elements of y
    :param x: a tensor x
    :param y: a tensor y
    :return: 1 if all elements of x are equal to all elements of y, 0 otherwise
    """
    cmp = x - y
    cmp = cmp.tensor.flatten().sum(axis=0)
    if cmp != 0:
        raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")
    return 1