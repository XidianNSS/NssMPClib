#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
import os

import torch
from pyasn1_modules.rfc5753 import x9_63_scheme

from nssmpc.config import param_path, DEBUG_LEVEL, data_type, DEVICE, DTYPE
from nssmpc.infra.mpc.aux_parameter.param_provider import BaseParamProvider
from nssmpc.infra.mpc.aux_parameter.parameter import Parameter
from nssmpc.infra.mpc.party import Party, PartyCtx
from nssmpc.infra.tensor import RingTensor
from nssmpc.infra.utils.common import align_shape
from nssmpc.infra.utils.cuda import cuda_matmul
from nssmpc.primitives.secret_sharing import ReplicatedSecretSharing
from nssmpc.protocols.honest_majority_3pc.base import hm3pc_open
from nssmpc.protocols.honest_majority_3pc import hm3pc_check_zero
from nssmpc.protocols.semi_honest_3pc.multiplication import sh3pc_mul_with_out_trunc, sh3pc_matmul_with_out_trunc
from nssmpc.protocols.honest_majority_3pc.truncation import hm3pc_truncate_aby3


def hm3pc_mul(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing, party: Party = None) -> ReplicatedSecretSharing:
    """Performs secure element-wise multiplication of RSS shares with truncation.

    Args:
        x: First RSS input.
        y: Second RSS input.
        party: The party instance. Defaults to current context.

    Returns:
        ReplicatedSecretSharing: Truncated product of x and y.

    Examples:
        >>> res = hm3pc_mul(x, x)
    """
    if party is None:
        party = PartyCtx.get()
    ori_type = x.dtype
    res = sh3pc_mul_with_out_trunc(x, y)

    a, b, c = party.get_param(RssMulTriples, res.numel())  # TODO: need fix, get triples based on x.shape and y.shape
    x_hat = x.clone()
    y_hat = y.clone()

    a.dtype = b.dtype = c.dtype = x_hat.dtype = y_hat.dtype = 'int'
    e = x_hat + a
    f = y_hat + b
    e_and_f = x.__class__.cat([e.flatten(), f.flatten()], dim=0)
    common_e_f = hm3pc_open(e_and_f)
    e = common_e_f[:x.numel()].reshape(x.shape)
    f = common_e_f[x.numel():].reshape(y.shape)

    check = -c + b * e + a * f - e * f

    hm3pc_check_zero(res + check)

    if ori_type == 'float':
        res = hm3pc_truncate_aby3(res)

    return res


def hm3pc_matmul(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing,
                 party: Party = None) -> ReplicatedSecretSharing:
    """Performs secure matrix multiplication of RSS shares with truncation.

    Args:
        x: First RSS matrix.
        y: Second RSS matrix.
        party: The party instance. Defaults to current context.

    Returns:
        ReplicatedSecretSharing: Truncated matrix product of x and y.

    Examples:
        >>> res = hm3pc_matmul(x, x)
    """
    if party is None:
        party = PartyCtx.get()

    x_shape, y_shape = align_shape(x.shape, y.shape)

    a, b, c = party.get_param(RssMatmulTriples, 1, tag=f'{x_shape}_{y_shape}')

    ori_type = x.dtype
    res = sh3pc_matmul_with_out_trunc(x, y)

    x_hat = x.clone()
    y_hat = y.clone()

    a.dtype = b.dtype = c.dtype = x_hat.dtype = y_hat.dtype = 'int'
    e = x_hat + a
    f = y_hat + b

    e_and_f = x.__class__.cat([e.flatten(), f.flatten()], dim=0)
    common_e_f = hm3pc_open(e_and_f)
    e = common_e_f[:x.numel()].reshape(x.shape)
    f = common_e_f[x.numel():].reshape(y.shape)

    mat_1 = ReplicatedSecretSharing([e @ b.item[0], e @ b.item[1]])
    mat_2 = ReplicatedSecretSharing([a.item[0] @ f, a.item[1] @ f])

    check = -c + mat_1 + mat_2 - e @ f
    hm3pc_check_zero(res + check)

    if ori_type == 'float':
        res = hm3pc_truncate_aby3(res)
    return res


class RssMulTriples(Parameter):
    """Parameter class for RSS multiplication triples."""

    def __init__(self, a=None, b=None, c=None):
        """Initializes empty multiplication triple components."""
        self.a = a
        self.b = b
        self.c = c
        self.size = 0

    def __iter__(self):
        """Allows iteration over triple components (a, b, c)."""
        return iter((self.a, self.b, self.c))

    @staticmethod
    def gen(num_of_triples: int):
        """Generates multiplicative Beaver triples using a trusted third party.

        Args:
            num_of_triples: Number of triples to generate.

        Returns:
            list[RssMulTriples]: List of generated triples for each party.

        Examples:
            >>> triples = RssMulTriples.gen(10)
        """
        a = RingTensor.random([num_of_triples])
        b = RingTensor.random([num_of_triples])
        c = a * b
        a_list = ReplicatedSecretSharing.share(a)
        b_list = ReplicatedSecretSharing.share(b)
        c_list = ReplicatedSecretSharing.share(c)
        triples = []
        for i in range(3):
            triples.append(RssMulTriples(a_list[i].to('cpu'), b_list[i].to('cpu'), c_list[i].to('cpu')))
            triples[i].size = num_of_triples
        return triples


class RssMatmulTriples(Parameter):
    def __init__(self, a=None, b=None, c=None):
        """Initializes empty multiplication triple components."""
        self.a = a
        self.b = b
        self.c = c
        self.size = 0

    def __iter__(self):
        """Allows iteration over triple components (a, b, c)."""
        return iter((self.a, self.b, self.c))

    @staticmethod
    def gen(num_of_triples: int, x_shape: list[int] = None, y_shape: list[int] = None, num_of_party=3):
        """Generates matrix multiplication triples using a trusted third party.

        Args:
            num_of_triples: Number of triples.
            x_shape: Shape of matrix X. Defaults to None.
            y_shape: Shape of matrix Y. Defaults to None.
            num_of_party (int, optional): Number of parties. Defaults to 3.

        Returns:
            list[RssMatmulTriples]: List of generated triples for each party.

        Examples:
            >>> triples = RssMatmulTriples.gen(10, [5, 5], [5, 5])
        """
        x_shape = [num_of_triples] + list(x_shape)
        y_shape = [num_of_triples] + list(y_shape)
        a = RingTensor.random(x_shape)
        b = RingTensor.random(y_shape)
        if a.device == 'cpu':
            c = a @ b
        else:
            c = cuda_matmul(a.tensor, b.tensor)
            c = RingTensor.convert_to_ring(c)
        a_list = ReplicatedSecretSharing.share(a)
        b_list = ReplicatedSecretSharing.share(b)
        c_list = ReplicatedSecretSharing.share(c)
        triples = []
        for i in range(num_of_party):
            triples.append(RssMatmulTriples())
            triples[i].a = a_list[i].to('cpu')
            triples[i].b = b_list[i].to('cpu')
            triples[i].c = c_list[i].to('cpu')
        return triples

    @classmethod
    def gen_and_save(cls, num: int, shape_x: list[int], shape_y: list[int],
                     type_of_generation: str = 'TTP', party=None):
        """Generates and saves matrix multiplication triples to disk.

        Args:
            num: Number of triples.
            shape_x: Shape of matrix X.
            shape_y: Shape of matrix Y.
            type_of_generation: Generation method ('TTP' or 'HE'). Defaults to 'TTP'.
            party (Party, optional): Party instance for HE generation. Defaults to None.

        Examples:
            >>> RssMatmulTriples.gen_and_save(100,[5, 5],[5, 5])
        """
        shape_x, shape_y = align_shape(shape_x, shape_y)
        triples = cls.gen(num, shape_x, shape_y)
        if type_of_generation == 'TTP':
            file_path = param_path + f"RssMatmulTriples/"
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            for i in range(3):
                file_name = f"p{i}_RssMatmulTriples_{list(shape_x)}_{list(shape_y)}.pth"
                triples[i].save(file_path, file_name)
