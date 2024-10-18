#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.config import EXP_ITER


def secure_exp(x):
    """
    Approximates the exponential function using an iterative limit approximation.

    This approximation is based on the following mathematical expression:

        :math:`\\exp(x) = \\lim_{n \to \\infty} \\left(1 + \\frac{x}{n}\\right)^n`

    The function approximates the exponential by choosing `n = 2^d` where `d` is the number of
    iterations defined by the constant `EXP_ITER`. It computes the value of :math:`\\left(1 + \\frac{x}{n}\\right)`
    and then squares it `d` times to approximate :math:`e^x`.

    :param x: The input ASS for which the exponential function is to be approximated.
    :type x: ArithmeticSecretSharing
    :return: The approximation of :math:`e^x` based on the iterative limit.
    :rtype: ArithmeticSecretSharing
    """
    result = x / (2 ** EXP_ITER) + 1

    for _ in range(EXP_ITER):
        result = result * result

    return result
