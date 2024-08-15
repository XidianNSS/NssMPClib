from NssMPC import RingTensor
from NssMPC.crypto.aux_parameter import AssMulTriples, MatmulTriples


def beaver_mul(x, y):
    """
    ASS multiplication using beaver triples
    Returns:
        ArithmeticSharedRingTensor: multiplication result
    """
    shape = x.shape if x.numel() > y.numel() else y.shape
    x = x.expand(shape).flatten()
    y = y.expand(shape).flatten()
    a, b, c = x.party.get_param(AssMulTriples, x.numel())
    a.dtype = b.dtype = c.dtype = x.dtype
    e = x - a
    f = y - b

    e_and_f = x.__class__.cat([e, f], dim=0)
    common_e_f = e_and_f.restore()
    length = common_e_f.shape[0] // 2
    common_e = common_e_f[:length]
    common_f = common_e_f[length:]

    res1 = RingTensor.mul(common_e, common_f) * x.party.party_id
    res2 = RingTensor.mul(a.item, common_f)
    res3 = RingTensor.mul(common_e, b.item)
    res = res1 + res2 + res3 + c.item

    res = x.__class__(res, x.party)
    return res.reshape(shape)


def secure_matmul(x, y):
    """
    ASS matrix multiplication using beaver triples

    Returns:
        ArithmeticSharedRingTensor: matrix multiplication result
    """
    a_matrix, b_matrix, c_matrix = x.party.get_param(MatmulTriples, x.shape, y.shape)

    e = x - a_matrix
    f = y - b_matrix

    e_and_f = x.__class__.cat([e.flatten(), f.flatten()], dim=0)
    common_e_f = e_and_f.restore()
    common_e = common_e_f[:x.numel()].reshape(x.shape)
    common_f = common_e_f[x.numel():].reshape(y.shape)

    res1 = RingTensor.matmul(common_e, common_f)
    res2 = RingTensor.matmul(common_e, b_matrix.item)
    res3 = RingTensor.matmul(a_matrix.item, common_f)

    res = res1 * x.party.party_id + res2 + res3 + c_matrix.item

    res = x.__class__(res, x.party)

    return res
