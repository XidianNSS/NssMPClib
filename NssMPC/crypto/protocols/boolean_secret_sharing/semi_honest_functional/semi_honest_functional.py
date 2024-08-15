from NssMPC.crypto.aux_parameter.beaver_triples import BooleanTriples


def beaver_and(x, y):
    """
    ASS multiplication using beaver triples
    Returns:
        ArithmeticSharedRingTensor: multiplication result
    """
    shape = x.shape if x.numel() > y.numel() else y.shape
    x = x.expand(shape).flatten()
    y = y.expand(shape).flatten()
    a, b, c = x.party.get_param(BooleanTriples, x.numel())
    a.dtype = b.dtype = c.dtype = x.dtype
    e = x ^ a
    f = y ^ b

    e_and_f = x.__class__.cat([e, f], dim=0)
    common_e_f = e_and_f.restore()
    length = common_e_f.shape[0] // 2
    common_e = common_e_f[:length]
    common_f = common_e_f[length:]

    res1 = common_e & common_f & x.party.party_id
    res2 = a.item & common_f
    res3 = common_e & b.item
    res = res1 ^ res2 ^ res3 ^ c.item

    res = x.__class__(res, x.party)
    return res.view(shape)
