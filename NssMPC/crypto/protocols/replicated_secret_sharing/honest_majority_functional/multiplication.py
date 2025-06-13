#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.aux_parameter import RssMulTriples, RssMatmulTriples
from NssMPC.crypto.protocols.replicated_secret_sharing.semi_honest_functional import mul_with_out_trunc, \
    matmul_with_out_trunc
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.base import check_zero, open
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.truncate import truncate


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
    ori_type = x.dtype
    res = mul_with_out_trunc(x, y)

    a, b, c = PartyRuntime.party.get_param(RssMulTriples,
                                           res.numel())  # TODO: need fix, get triples based on x.shape and y.shape
    x_hat = x.clone()
    y_hat = y.clone()

    a.dtype = b.dtype = c.dtype = x_hat.dtype = y_hat.dtype = 'int'
    e = x_hat + a
    f = y_hat + b
    e_and_f = x.__class__.cat([e.flatten(), f.flatten()], dim=0)
    common_e_f = open(e_and_f)
    e = common_e_f[:x.numel()].reshape(x.shape)
    f = common_e_f[x.numel():].reshape(y.shape)

    check = -c + b * e + a * f - e * f

    check_zero(res + check)

    if ori_type == 'float':
        res = truncate(res)

    return res


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
    a, b, c = PartyRuntime.party.get_param(RssMatmulTriples, x.shape, y.shape)

    ori_type = x.dtype
    res = matmul_with_out_trunc(x, y)

    x_hat = x.clone()
    y_hat = y.clone()

    a.dtype = b.dtype = c.dtype = x_hat.dtype = y_hat.dtype = 'int'
    e = x_hat + a
    f = y_hat + b

    e_and_f = x.__class__.cat([e.flatten(), f.flatten()], dim=0)
    common_e_f = open(e_and_f)
    e = common_e_f[:x.numel()].reshape(x.shape)
    f = common_e_f[x.numel():].reshape(y.shape)

    from NssMPC.crypto.primitives.arithmetic_secret_sharing import ReplicatedSecretSharing
    mat_1 = ReplicatedSecretSharing([e @ b.item[0], e @ b.item[1]])
    mat_2 = ReplicatedSecretSharing([a.item[0] @ f, a.item[1] @ f])

    check = -c + mat_1 + mat_2 - e @ f
    check_zero(res + check)

    if ori_type == 'float':
        res = truncate(res)
    return res
