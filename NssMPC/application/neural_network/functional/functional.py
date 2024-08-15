from NssMPC.crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing


def img2col_for_conv(img, k_size: int, stride: int):
    """Tensor deformation method for convolution layers
    """
    col, batch, out_size, _ = img.item.img2col(k_size, stride)
    return img.__class__(col.reshape((batch, -1, out_size)), img.party)


def img2col_for_pool(img, k_size: int, stride: int):
    """Tensor deformation method for pooling layers
    """
    col, batch, out_size, channel = img.item.img2col(k_size, stride)
    return img.__class__(col.reshape((batch, channel, -1, out_size)), img.party)


def torch2share(param, share_type, dtype, party):
    if share_type == ArithmeticSecretSharing:
        return ArithmeticSecretSharing(RingTensor(param, dtype), party)
    elif share_type == ReplicatedSecretSharing:
        return ReplicatedSecretSharing([RingTensor(param[0], dtype), RingTensor(param[1], dtype)], party)
    else:
        raise ValueError("Unknown share type!")
