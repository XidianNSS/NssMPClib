from NssMPC.common.ring.ring_tensor import RingTensor


def mul_with_out_trunc(x, y):
    z_shared = RingTensor.mul(x.item[0], y.item[0]) + RingTensor.mul(x.item[0], y.item[1]) + RingTensor.mul(x.item[1],
                                                                                                            y.item[0])
    from NssMPC.crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
    result = ReplicatedSecretSharing.reshare(z_shared, x.party)
    return result


def matmul_with_out_trunc(x, y):
    t_i = \
        RingTensor.matmul(x.item[0], y.item[0]) + \
        RingTensor.matmul(x.item[0], y.item[1]) + \
        RingTensor.matmul(x.item[1], y.item[0])
    from NssMPC.crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
    result = ReplicatedSecretSharing.reshare(t_i, x.party)
    return result
