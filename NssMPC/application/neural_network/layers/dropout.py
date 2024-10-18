"""
Dropout is a regularization technique that is widely used in neural network training to prevent overfitting. Its
basic idea is to randomly "drop out" a portion of neurons during each training iteration, so that the network is not
dependent on certain specific neurons during training and thus improves the model's generalization ability.
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch


class SecDropout(torch.nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        Initialize the SecDropout. Call the constructor of the parent class (torch.nn.Module).

        :param p: The probability of discarding, assumed to be 0.5 by default, indicates that there is a 50% chance of setting the output of certain neurons to zero in each forward propagation.
        :type p: float
        :param inplace: If it is True, the operation will be performed in place, which may save memory.
        :type inplace: bool
        """
        super(SecDropout, self).__init__()
        pass

    def forward(self, x):
        """
        The forward propagation process.

        :param x: Input tensor, typically coming from the output of the previous layer.
        :type x: ArithmeticSecretSharing
        :return: Simply return the input ``x`` to indicate that no dropout operation has been performed yet.
        :rtype: ArithmeticSecretSharing
        """
        return x


def _SecDropout(x):
    """
    This is a simple utility function. It simply returns the input ``x`` without doing any processing on it.

    :param x: Input tensor
    :type x: ArithmeticSecretSharing
    :return: Simply return the input x to indicate that no dropout operation has been performed yet.
    :rtype: ArithmeticSecretSharing
    """
    return x
