from common.tensor.ring_tensor import RingTensor
from crypto.primitives.beaver.beaver_triples import BeaverTriples
from crypto.primitives.beaver.matrix_triples import MatrixTriples


def beaver_mul(x, y):
    """
    ASS multiplication using beaver triples

    Returns:
        ArithmeticSharedRingTensor: multiplication result
    """
    shape = x.shape
    x = x.flatten()
    y = y.flatten()
    a, b, c = x.party.get_param(BeaverTriples, x.numel())
    a.dtype = b.dtype = c.dtype = x.dtype
    e = x - a
    f = y - b

    common_e = e.restore()
    common_f = f.restore()

    res1 = RingTensor.mul(common_e, common_f) * x.party.party_id
    res2 = RingTensor.mul(a, common_f)
    res3 = RingTensor.mul(common_e, b)
    res = res1 + res2 + res3 + c

    res = x.__class__(res, x.party)

    return res.view(shape)


def secure_matmul(x, y):
    """
    ASS matrix multiplication using beaver triples

    Returns:
        ArithmeticSharedRingTensor: matrix multiplication result
    """
    a_matrix, b_matrix, c_matrix = x.party.get_param(MatrixTriples, x.shape, y.shape)

    e = x - a_matrix
    f = y - b_matrix

    common_e = e.restore()
    common_f = f.restore()

    res1 = RingTensor.matmul(common_e, common_f)
    res2 = RingTensor.matmul(common_e, b_matrix)
    res3 = RingTensor.matmul(a_matrix, common_f)

    res = res1 * x.party.party_id + res2 + res3 + c_matrix

    res = x.__class__(res, x.party)

    return res


def secure_div(x, y):
    """
    Implement ASS division protocols using iterative method
    Correct only if y > 0 and BIT_LEN = 64
    TODO： support other bit length and y < 0.

    Args:
       x: dividend
       y: divisor

    Returns:
       quotient
    """

    powers = [(2 ** (i - 1)) * 1.0 for i in range(-15, y.ring_tensor.bit_len - 48)]
    powers = RingTensor.convert_to_ring(powers).to(x.device)

    for _ in range(len(y.shape)):
        powers = powers.unsqueeze(-1)

    powers = x.__class__(powers, party=x.party)

    k = (y.unsqueeze(0) >= powers).sum(dim=0)

    clear_k = k.restore().convert_to_real_field() - 15

    k_range = 2 ** clear_k

    ring_k_range = RingTensor.convert_to_ring(k_range)

    a = x / ring_k_range
    b = y / ring_k_range

    w = b * (-2) + 2.9142

    e0 = -(b * w) + 1

    e1 = e0 * e0
    res = a * w * (e0 + 1) * (e1 + 1)

    return res
