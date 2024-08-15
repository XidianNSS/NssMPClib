from NssMPC.crypto.aux_parameter import RssMulTriples, RssMatmulTriples
from NssMPC.crypto.protocols.replicated_secret_sharing.semi_honest_functional import mul_with_out_trunc, \
    matmul_with_out_trunc
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.base import check_zero, open
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.truncation import truncate


def v_mul(x, y):
    """
    Multiplication of two RSS sharings ⟨x⟩ and ⟨y⟩
    :param x: an RSS sharing ⟨x⟩
    :param y: an RSS sharing ⟨y⟩
    :return: an RSS sharing ⟨x*y⟩
    """
    shape = x.shape if x.numel() > y.numel() else y.shape
    x = x.expand(shape).flatten()
    y = y.expand(shape).flatten()
    ori_type = x.dtype
    res = mul_with_out_trunc(x, y)
    # print("单纯乘法：", res.restore())

    # a, b, c = x.party.providers[AssMulTriples].get_parameters(res.numel())
    a, b, c = x.party.get_param(RssMulTriples, res.numel())
    x_hat = x.clone()
    y_hat = y.clone()
    # print("c", c.restore())
    # c_tmp = mul_with_out_trunc(a, b)
    # print("c tmp", c_tmp.restore())

    # a = a.reshape(x.shape)
    # b = b.reshape(y.shape)
    # c = c.reshape(res.shape)
    # todo 临时方法，缺少原理论证
    a.dtype = b.dtype = c.dtype = x_hat.dtype = y_hat.dtype = 'int'
    e = x_hat + a
    f = y_hat + b
    e = open(e)
    f = open(f)
    # tmp1 = mul_with_out_trunc(b, e)
    check = -c + b * e + a * f - e * f
    # print("辅助参数乘法：", check.restore())
    check_zero(res + check)

    if ori_type == 'float':
        res = truncate(res)

    return res.reshape(shape)


def v_matmul(x, y):
    """
    Matrix Multiplication of two RSS sharings ⟨x⟩ and ⟨y⟩
    :param x: an RSS sharing ⟨x⟩
    :param y: an RSS sharing ⟨y⟩
    :aux_params: verify params
    :return: an RSS sharing ⟨x@y⟩

    """
    a, b, c = x.party.get_param(RssMatmulTriples, x.shape, y.shape)
    a.party = x.party
    b.party = x.party
    c.party = x.party

    ori_type = x.dtype
    res = matmul_with_out_trunc(x, y)

    x_hat = x.clone()
    y_hat = y.clone()
    # print("ori_res", res.restore())
    a.dtype = b.dtype = c.dtype = x_hat.dtype = y_hat.dtype = 'int'
    e = x_hat + a
    f = y_hat + b
    e = open(e)
    f = open(f)
    from NssMPC.crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
    mat_1 = ReplicatedSecretSharing([e @ b.item[0], e @ b.item[1]], x.party)
    mat_2 = ReplicatedSecretSharing([a.item[0] @ f, a.item[1] @ f], x.party)

    check = -c + mat_1 + mat_2 - e @ f
    check_zero(res + check)
    # print(ori_type)
    if ori_type == 'float':
        res = truncate(res)
    return res
