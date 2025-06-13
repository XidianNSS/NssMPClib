#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.crypto.primitives.arithmetic_secret_sharing import ReplicatedSecretSharing
from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.crypto.primitives.arithmetic_secret_sharing import ArithmeticSecretSharing


def img2col_for_conv(img, k_size: int, stride: int):
    """
    Tensor deformation method for convolution layers.

    Calling ``img.item.img2col(k_size, stride)`` transforms the image into column format, and the returned value includes
    the column data, batch size, and output size.

    :param img: Input tensor image, usually with multiple dimensions.
    :type img: RingTensor or ArithmeticSecretSharing or ReplicatedSecretSharing
    :param k_size: The size of the convolutional kernel
    :type k_size: int
    :param stride: Step size of convolution
    :type stride: int
    :return: Input image tensor in column form
    :rtype: RingTensor or ArithmeticSecretSharing or ReplicatedSecretSharing
    """
    col, batch, out_size, _ = img.item.img2col(k_size, stride)
    return img.__class__(col.reshape((batch, -1, out_size)))


def img2col_for_pool(img, k_size: int, stride: int):
    """
    Tensor deformation method for pooling layers.

    Similar to convolution, the transformation is performed by calling ``img.item.img2col(k_size, stride)``.

    :param img: Input tensor image, usually with multiple dimensions.
    :type img: RingTensor or ArithmeticSecretSharing or ReplicatedSecretSharing
    :param k_size: The size of the convolutional kernel
    :type k_size: int
    :param stride: Step size of convolution
    :type stride: int
    :return: Input image tensor in column form
    :rtype: RingTensor or ArithmeticSecretSharing or ReplicatedSecretSharing
    """
    col, batch, out_size, channel = img.item.img2col(k_size, stride)
    return img.__class__(col.reshape((batch, channel, -1, out_size)))


def torch2share(param, share_type, dtype):
    """
    Convert the parameters into a shared form so they can be used in secure multi-party computation.

    Based on the ``share_type``, select the appropriate sharing method:
        * For :class:`~NssMPC.crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing.ArithmeticSecretSharing` : Create a :class:`~NssMPC.common.ring.ring_tensor.RingTensor` instance and wrap it;
        * For :class:`~NssMPC.crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing.ReplicatedSecretSharing` : Wrap the parameters as 2 :class:`~NssMPC.common.ring.ring_tensor.RingTensor` instances.

    :param param: The input parameter can be any tensor
    :type param: torch.Tensor
    :param share_type: Share type, which determines how input parameters are handled
    :type share_type: ArithmeticSecretSharing or ReplicatedSecretSharing
    :param dtype: Data type, specifying the type of the tensor.
    :type dtype: str or dtype
    :return: Parameters after conversion form
    :rtype: ArithmeticSecretSharing or ReplicatedSecretSharing
    :raises ValueError: If ``share_type`` is invalid
    """
    if share_type == ArithmeticSecretSharing:
        return ArithmeticSecretSharing(RingTensor(param, dtype))
    elif share_type == ReplicatedSecretSharing:
        return ReplicatedSecretSharing([RingTensor(param[0], dtype), RingTensor(param[1], dtype)])
    else:
        raise ValueError("Unknown share type!")
