from NssMPC.common.ring.ring_tensor import RingTensor


def rand(shape, party):
    num_of_value = 1
    for d in shape:
        num_of_value *= d
    r_0 = party.prg_0.random(num_of_value)
    r_1 = party.prg_1.random(num_of_value)
    from NssMPC.crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
    r = ReplicatedSecretSharing([r_0, r_1], party)
    r = r.reshape(shape)
    return r


def rand_like(x, party):
    r = rand(x.shape, party)
    # r = r.reshape(x.shape)
    if isinstance(x, RingTensor):
        r.dtype = x.dtype
    return r
