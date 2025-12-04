#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.infra.tensor import RingTensor
from NssMPC.primitives.secret_sharing import AdditiveSecretSharing, ReplicatedSecretSharing


def img2col_for_conv(img, k_size: int, stride: int):
    """Tensor deformation method for convolution layers.

    Calling ``img.item.img2col(k_size, stride)`` transforms the image into column format, and the returned value includes
    the column data, batch size, and output size.

    Args:
        img (RingTensor or AdditiveSecretSharing or ReplicatedSecretSharing): Input tensor image, usually with multiple dimensions.
        k_size: The size of the convolutional kernel.
        stride: Step size of convolution.

    Returns:
        RingTensor or ArithmeticSecretSharing or ReplicatedSecretSharing: Input image tensor in column form.

    Examples:
        >>> col_img = img2col_for_conv(img, k_size=3, stride=1)
    """
    col, batch, out_size, _ = img.item.img2col(k_size, stride)
    return img.__class__(col.reshape((batch, -1, out_size)))


def img2col_for_pool(img, k_size: int, stride: int):
    """Tensor deformation method for pooling layers.

    Similar to convolution, the transformation is performed by calling ``img.item.img2col(k_size, stride)``.

    Args:
        img (RingTensor or AdditiveSecretSharing or ReplicatedSecretSharing): Input tensor image, usually with multiple dimensions.
        k_size: The size of the convolutional kernel.
        stride: Step size of convolution.

    Returns:
        RingTensor or ArithmeticSecretSharing or ReplicatedSecretSharing: Input image tensor in column form.

    Examples:
        >>> col_img = img2col_for_pool(img, k_size=2, stride=2)
    """
    col, batch, out_size, channel = img.item.img2col(k_size, stride)
    return img.__class__(col.reshape((batch, channel, -1, out_size)))


def torch2share(param, share_type, dtype):
    """Convert the parameters into a shared form so they can be used in secure multi-party computation.

    Based on the ``share_type``, select the appropriate sharing method:
        * For :class:`~NssMPC.crypto.primitives.secret_sharing.secret_sharing.ArithmeticSecretSharing` : Create a :class:`~NssMPC.infra.ring.ring_tensor.RingTensor` instance and wrap it;
        * For :class:`~NssMPC.crypto.primitives.secret_sharing.replicated_secret_sharing.ReplicatedSecretSharing` : Wrap the parameters as 2 :class:`~NssMPC.infra.ring.ring_tensor.RingTensor` instances.

    Args:
        param (torch.Tensor): The input parameter can be any tensor.
        share_type (type): Share type, which determines how input parameters are handled (e.g. AdditiveSecretSharing or ReplicatedSecretSharing).
        dtype (str or dtype): Data type, specifying the type of the tensor.

    Returns:
        ArithmeticSecretSharing or ReplicatedSecretSharing: Parameters after conversion form.

    Raises:
        ValueError: If ``share_type`` is invalid.

    Examples:
        >>> shared_param = torch2share(param, AdditiveSecretSharing, 'int64')
    """
    if share_type == AdditiveSecretSharing:
        return AdditiveSecretSharing(RingTensor(param, dtype))
    elif share_type == ReplicatedSecretSharing:
        return ReplicatedSecretSharing([RingTensor(param[0], dtype), RingTensor(param[1], dtype)])
    else:
        raise ValueError("Unknown share type!")
