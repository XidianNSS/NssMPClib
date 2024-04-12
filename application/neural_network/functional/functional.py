from functools import singledispatch

from crypto.primitives.arithmetic_secret_sharing.arithmetic_shared_ring_tensor import ArithmeticSharedRingTensor


@singledispatch
def img2col_for_conv(img, k_size: int, stride: int):
    """Tensor deformation method for convolution layers
    """
    raise TypeError("Unsupported type")


@img2col_for_conv.register
def _(ass_img: ArithmeticSharedRingTensor, k_size: int, stride: int):
    col, batch, out_size, _ = ass_img.ring_tensor.img2col(k_size, stride)
    return ArithmeticSharedRingTensor(col.reshape((batch, -1, out_size)), ass_img.party)


@singledispatch
def img2col_for_pool(img, k_size: int, stride: int):
    """Tensor deformation method for pooling layers
    """
    raise TypeError("Unsupported type")


@img2col_for_pool.register
def _(ass_img: ArithmeticSharedRingTensor, k_size: int, stride: int):
    col, batch, out_size, channel = ass_img.ring_tensor.img2col(k_size, stride)
    return ArithmeticSharedRingTensor(col.reshape((batch, channel, -1, out_size)), ass_img.party)
