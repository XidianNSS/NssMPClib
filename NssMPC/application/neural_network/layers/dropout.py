"""
Dropout is a regularization technique that is widely used in neural communication training to prevent overfitting. Its
basic idea is to randomly "drop out" a portion of neurons during each training iteration, so that the communication is not
dependent on certain specific neurons during training and thus improves the model's generalization ability.
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch


class SecDropout(torch.nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        Initializes the SecDropout.

        Args:
            p: The probability of discarding. Defaults to 0.5.
            inplace: If True, the operation will be performed in place. Defaults to False.

        Examples:
            >>> dropout = SecDropout(p=0.5)
        """
        super(SecDropout, self).__init__()
        pass

    def forward(self, x):
        """
        The forward propagation process.

        Args:
            x (ArithmeticSecretSharing): Input tensor.

        Returns:
            ArithmeticSecretSharing: Returns the input `x` (currently no dropout operation performed).

        Examples:
            >>> output = dropout(input_tensor)
        """
        return x


def _SecDropout(x):
    """
    A simple utility function that returns the input `x`.

    Args:
        x (ArithmeticSecretSharing): Input tensor.

    Returns:
        ArithmeticSecretSharing: The input tensor.

    Examples:
        >>> output = _SecDropout(input_tensor)
    """
    return x
