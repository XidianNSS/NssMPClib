from NssMPC.config import EXP_ITER


def secure_exp(x):
    """
    Approximates the exponential function using a limit approximation:

    .. math::

        exp(x) = \lim_{n \\rightarrow \\infty} (1 + x / n) ^ n

    Here we compute exp by choosing n = 2 ** d for some large d equal to
    `iterations`. We then compute (1 + x / n) once and square `d` times.

    :param x:
    :return:
    """
    result = x / (2 ** EXP_ITER) + 1

    for _ in range(EXP_ITER):
        result = result * result

    return result
