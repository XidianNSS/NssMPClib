#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.crypto.aux_parameter import RssMulTriples, RssMatmulTriples
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.base import check_zero, open
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.truncate import truncate
from NssMPC.crypto.protocols.replicated_secret_sharing.semi_honest_functional import mul_with_out_trunc, \
    matmul_with_out_trunc


def v_mul(x, y):
    """
    Multiplication of two RSS sharings ⟨x⟩ and ⟨y⟩.

    :param x: An RSS sharing ⟨x⟩
    :type x: ReplicatedSecretSharing
    :param y: An RSS sharing ⟨y⟩
    :type y: ReplicatedSecretSharing
    :return: An RSS sharing ⟨x*y⟩
    :rtype: ReplicatedSecretSharing
    """
    shape = x.shape if x.numel() > y.numel() else y.shape
    x = x.expand(shape).flatten()
    y = y.expand(shape).flatten()
    ori_type = x.dtype
    res = mul_with_out_trunc(x, y)

    a, b, c = x.party.get_param(RssMulTriples, res.numel())
    x_hat = x.clone()
    y_hat = y.clone()

    a.dtype = b.dtype = c.dtype = x_hat.dtype = y_hat.dtype = 'int'
    e = x_hat + a
    f = y_hat + b
    e = open(e)
    f = open(f)

    check = -c + b * e + a * f - e * f

    check_zero(res + check)

    if ori_type == 'float':
        res = truncate(res)

    return res.reshape(shape)


def v_matmul(x, y):
    """
    Matrix Multiplication of two RSS sharings ⟨x⟩ and ⟨y⟩

    :param x: An RSS sharing ⟨x⟩
    :type x: ReplicatedSecretSharing
    :param y: An RSS sharing ⟨y⟩
    :type y: ReplicatedSecretSharing
    :return: An RSS sharing ⟨x@y⟩
    :rtype: ReplicatedSecretSharing

    """
    a, b, c = x.party.get_param(RssMatmulTriples, x.shape, y.shape)
    a.party = x.party
    b.party = x.party
    c.party = x.party

    ori_type = x.dtype
    res = matmul_with_out_trunc(x, y)

    x_hat = x.clone()
    y_hat = y.clone()

    a.dtype = b.dtype = c.dtype = x_hat.dtype = y_hat.dtype = 'int'
    e = x_hat + a
    f = y_hat + b
    e = open(e)
    f = open(f)
    from NssMPC.crypto.primitives.arithmetic_secret_sharing import ReplicatedSecretSharing
    mat_1 = ReplicatedSecretSharing([e @ b.item[0], e @ b.item[1]], x.party)
    mat_2 = ReplicatedSecretSharing([a.item[0] @ f, a.item[1] @ f], x.party)

    check = -c + mat_1 + mat_2 - e @ f
    check_zero(res + check)
    # print(ori_type)
    if ori_type == 'float':
        res = truncate(res)
    return res
