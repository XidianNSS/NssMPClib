#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
import os

import torch

from NssMPC.config import param_path, DEBUG_LEVEL, data_type, DEVICE, DTYPE
from NssMPC.infra.mpc.aux_parameter.param_provider import FixedShapeProvider
from NssMPC.infra.mpc.aux_parameter.parameter import Parameter
from NssMPC.infra.mpc.party import Party, PartyCtx
from NssMPC.infra.tensor import RingTensor
from NssMPC.infra.utils.cuda_utils import cuda_matmul
from NssMPC.primitives.secret_sharing import ReplicatedSecretSharing
from NssMPC.protocols.honest_majority_3pc.base import open
from NssMPC.protocols.honest_majority_3pc.mac_check import check_zero
from NssMPC.protocols.semi_honest_3pc.multiplication import mul_with_out_trunc, matmul_with_out_trunc
from NssMPC.protocols.semi_honest_3pc.truncate import truncate


def v_mul(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing, party: Party = None) -> ReplicatedSecretSharing:
    """Performs secure element-wise multiplication of RSS shares with truncation.

    Args:
        x: First RSS input.
        y: Second RSS input.
        party: The party instance. Defaults to current context.

    Returns:
        ReplicatedSecretSharing: Truncated product of x and y.

    Examples:
        >>> res = v_mul(x, x)
    """
    if party is None:
        party = PartyCtx.get()
    ori_type = x.dtype
    res = mul_with_out_trunc(x, y)

    a, b, c = party.get_param(RssMulTriples, res.numel())  # TODO: need fix, get triples based on x.shape and y.shape
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


def v_matmul(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing,
             party: Party = None) -> ReplicatedSecretSharing:
    """Performs secure matrix multiplication of RSS shares with truncation.

    Args:
        x: First RSS matrix.
        y: Second RSS matrix.
        party: The party instance. Defaults to current context.

    Returns:
        ReplicatedSecretSharing: Truncated matrix product of x and y.

    Examples:
        >>> res = v_matmul(x, x)
    """
    if party is None:
        party = PartyCtx.get()
    a, b, c = party.get_param(RssMatmulTriples, x.shape, y.shape)

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

    mat_1 = ReplicatedSecretSharing([e @ b.item[0], e @ b.item[1]])
    mat_2 = ReplicatedSecretSharing([a.item[0] @ f, a.item[1] @ f])

    check = -c + mat_1 + mat_2 - e @ f
    check_zero(res + check)

    if ori_type == 'float':
        res = truncate(res)
    return res


class RssMulTriples(Parameter):
    """Parameter class for RSS multiplication triples."""

    def __init__(self):
        """Initializes empty multiplication triple components."""
        self.a = None
        self.b = None
        self.c = None
        self.size = 0

    def __iter__(self):
        """Allows iteration over triple components (a, b, c)."""
        return iter((self.a, self.b, self.c))

    def set_triples(self, a, b, c):
        """Sets the triple components.

        Args:
            a (RingTensor): First component.
            b (RingTensor): Second component.
            c (RingTensor): Third component (product).
        """
        self.a = a
        self.b = b
        self.c = c

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
            triples.append(RssMulTriples())
            triples[i].a = a_list[i].to('cpu')
            triples[i].b = b_list[i].to('cpu')
            triples[i].c = c_list[i].to('cpu')
            triples[i].size = num_of_triples
        return triples

    @classmethod
    def gen_and_save(cls, num: int, type_of_generation: str = 'TTP', party=None):
        """Generates and saves multiplicative Beaver triples to disk.

        Args:
            num: Number of triples.
            type_of_generation: Generation method ('TTP' or 'HE'). Defaults to 'TTP'.
            party (Party, optional): Party instance for HE generation. Defaults to None.

        Examples:
            >>> RssMulTriples.gen_and_save(100)
        """
        triples = cls.gen(num)
        if type_of_generation == 'TTP':
            file_path = f"{param_path}RssMulTriples/"
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            for i in range(3):
                file_name = f"RssMulTriples_{i}.pth"
                triples[i].save(file_path, file_name)


class RssMatmulTriples(RssMulTriples):
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
    def load_provider(cls, party):
        """Loads and attaches the matrix multiplication triple provider to the party.

        Args:
            party (Party): The party instance.

        Returns:
            RssMatrixBeaverProvider: The loaded provider.
        """
        # TODO check
        provider = RssMatrixBeaverProvider(party=party)
        party.append_provider(provider)
        return provider


class RssMatrixBeaverProvider(FixedShapeProvider):
    """Provider for RSS matrix multiplication Beaver triples."""

    def __init__(self, party=None, saved_name=None, param_tag=None, root_path=None):
        """Initializes the provider.

        Args:
            party (Party, optional): The party instance. Defaults to None.
            saved_name (str, optional): Filename for saved parameters. Defaults to None.
            param_tag (str, optional): Tag for parameter identification. Defaults to None.
            root_path (str, optional): Root directory for parameters. Defaults to None.
        """
        super().__init__(RssMatmulTriples, party, saved_name, param_tag, root_path)

    def load_param(self, x_shape=None, y_shape=None, num_of_party=2):
        """Loads matrix triples from disk.

        Args:
            x_shape (tuple, optional): Shape of matrix X. Defaults to None.
            y_shape (tuple, optional): Shape of matrix Y. Defaults to None.
            num_of_party (int, optional): Number of parties. Defaults to 2.

        Raises:
            Exception: If the parameter file is not found.
        """
        file_name = f"MatrixBeaverTriples_{self.party.party_id}_{list(x_shape)}_{list(y_shape)}.pth"
        file_path = param_path + f"BeaverTriples/{num_of_party}party/Matrix"
        try:
            self.param = RssMatmulTriples.load(file_path + file_name)  # TODO check
        except FileNotFoundError:
            raise Exception("Need generate matrix triples in this shape first!")

    def get_parameters(self, x_shape=None, y_shape=None):
        """Retrieves or generates matrix multiplication triples.

        Args:
            x_shape (tuple, optional): Shape of matrix X. Defaults to None.
            y_shape (tuple, optional): Shape of matrix Y. Defaults to None.

        Returns:
            RssMatmulTriples: The requested triples.
        """
        if DEBUG_LEVEL == 2:
            if f"{list(x_shape)}_{list(y_shape)}" not in self.param.keys():
                a = torch.ones(x_shape, dtype=data_type, device=DEVICE)
                b = torch.ones(y_shape, dtype=data_type, device=DEVICE)

                assert x_shape[-1] == y_shape[-2]
                c_shape = torch.broadcast_shapes(x_shape[:-2], y_shape[:-2]) + (x_shape[-2], y_shape[-1])
                c = torch.ones(c_shape, dtype=data_type, device=DEVICE) * x_shape[-1] * 3

                a_tensor = RingTensor(a, dtype=DTYPE).to(DEVICE)
                b_tensor = RingTensor(b, dtype=DTYPE).to(DEVICE)
                c_tensor = RingTensor(c, dtype=DTYPE).to(DEVICE)

                a = ReplicatedSecretSharing([a_tensor, a_tensor])
                b = ReplicatedSecretSharing([b_tensor, b_tensor])
                c = ReplicatedSecretSharing([c_tensor, c_tensor])

                self.param[f"{list(x_shape)}_{list(y_shape)}"] = RssMatmulTriples()
                self.param[f"{list(x_shape)}_{list(y_shape)}"].set_triples(a, b, c)
            return self.param[f"{list(x_shape)}_{list(y_shape)}"]
        else:
            mat_beaver = self.param[self.matrix_ptr].pop()
            self.matrix_ptr = (self.matrix_ptr + 1) % self.matrix_ptr_max
            return mat_beaver.to(DEVICE)
