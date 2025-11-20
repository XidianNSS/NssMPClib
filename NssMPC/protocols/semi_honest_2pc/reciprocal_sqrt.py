#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
from typing import Tuple

import torch

from NssMPC.config import SCALE, SCALE_BIT, data_type
from NssMPC.infra.mpc.param_provider.parameter import Parameter
from NssMPC.infra.mpc.party import Party, PartyCtx
from NssMPC.infra.tensor import RingTensor
from NssMPC.primitives.secret_sharing import AdditiveSecretSharing
from NssMPC.primitives.secret_sharing.function import SigmaDICF, SigmaDICFKey
from NssMPC.protocols.semi_honest_2pc.b2a import b2a
from NssMPC.protocols.semi_honest_2pc.look_up_table import LookUp
from NssMPC.protocols.semi_honest_2pc.look_up_table import LookUpKey


def secure_reciprocal_sqrt(x: AdditiveSecretSharing, party: Party = None) -> AdditiveSecretSharing:
    """
    Securely compute the reciprocal square root (1/sqrt(x)) of the input x.

    This function typically employs iterative methods (such as Newton-Raphson) to approximate the value
    while keeping the input secret.

    Note:
        Constraint: The result is guaranteed to be correct only if the input satisfies:
        0 < x < 2^{2f} (where f is the fixed-point scale).

    Args:
        x: The input Additive Secret Sharing value.
        party: The party instance managing the communication. Defaults to None.

    Returns:
        The secret-shared result of the reciprocal square root.

    Examples:
        >>> res = secure_reciprocal_sqrt(x)
    """
    if party is None:
        party = PartyCtx.get()
    key = party.get_param(ReciprocalSqrtKey, x.numel())
    return reciprocal_sqrt_eval(x, key)


def reciprocal_sqrt_eval(x: AdditiveSecretSharing, key: 'ReciprocalSqrtKey',
                         party: Party = None) -> AdditiveSecretSharing:
    """
    Securely evaluate the reciprocal square root using a pre-computed auxiliary key.

    This method accelerates the online computation phase by utilizing the provided key.

    Note:
        Constraint: The result is correct only if 0 < x < 2^{2f}.

    Args:
        x: The input Additive Secret Sharing value.
        key: The auxiliary parameter (key) used for the Reciprocal Square Root protocol.
        party: The party instance managing the communication. Defaults to None.

    Returns:
        The secret-shared result of the reciprocal square root.

    Examples:
        >>> res = reciprocal_sqrt_eval(x, key)
    """
    if party is None:
        party = PartyCtx.get()
    neg_exp2_table = key.neg_exp2_table
    rec_sqrt_table = key.rec_sqrt_table
    neg_exp2_look_up_key = key.neg_exp2_look_up_key
    rec_sqrt_lut_key = key.rec_sqrt_look_up_key
    sigma_key = key.sigma_key

    x_shape = x.shape
    x_shift = x.__class__(sigma_key.r_in) + x.flatten()
    x_shift = x_shift.restore()
    x_shift = x_shift.view(x_shape)

    x_minus_powers = [x_shift - (2 ** i) for i in range(1, 2 * SCALE_BIT + 1)]
    k = SigmaDICF.one_key_eval(x_minus_powers, sigma_key, party.party_id)
    k = b2a(k, party).sum(dim=0)
    neg_exp2_k = LookUp.eval(k, neg_exp2_look_up_key, neg_exp2_table)
    real_dtype = x.dtype
    x.dtype = 'int'
    u = x * neg_exp2_k * 128
    u.dtype = real_dtype
    u = u / (u.scale * u.scale)
    x.dtype = real_dtype

    m = u - (128 if SCALE == 1 else 128 / SCALE)

    p = (m * (2 ** 6) + k) * x.scale

    return LookUp.eval(p, rec_sqrt_lut_key, rec_sqrt_table)


class ReciprocalSqrtKey(Parameter):
    """
    A key management structure for dealing with square root reciprocal and negative index values.
    """

    def __init__(self):
        """
        Initializes the ReciprocalSqrtKey.

        Attributes:
            neg_exp2_look_up_key (LookUpKey): Store the search key for the negative exponent values.
            rec_sqrt_look_up_key (LookUpKey): Store the search key for the reciprocal square root.
            sigma_key (SigmaDICFKey): The Sigma protocol generates the key of DICF.
            neg_exp2_table (RingTensor): A lookup table for storing the negative exponent values.
            rec_sqrt_table (RingTensor): A lookup table for storing the reciprocal of square roots.
        """
        self.neg_exp2_look_up_key = LookUpKey()
        self.rec_sqrt_look_up_key = LookUpKey()
        self.sigma_key = SigmaDICFKey()
        self.neg_exp2_table = None
        self.rec_sqrt_table = None

    def __getitem__(self, item: int) -> 'ReciprocalSqrtKey':
        """
        Get item from the key.

        This method overrides __getitem__ to allow an instance of ReciprocalSqrtKey to be accessed via an index.
        It calls the __getitem__ method of the parent class and assigns neg_exp2_table and rec_sqrt_table to the
        returned key. This allows each key to access the same lookup table.

        Args:
            item: Index to find.

        Returns:
            Key corresponding to the index.

        Examples:
            >>> key_i = key[i]
        """
        key = super(ReciprocalSqrtKey, self).__getitem__(item)
        key.neg_exp2_table = self.neg_exp2_table
        key.rec_sqrt_table = self.rec_sqrt_table
        return key

    def __len__(self) -> int:
        """
        Get the length of the key.

        Returns:
            The length of sigma_key.
        """
        return len(self.sigma_key)

    @staticmethod
    def gen(num_of_keys: int, in_scale_bit: int = SCALE_BIT, out_scale_bit: int = SCALE_BIT) -> \
            Tuple['ReciprocalSqrtKey', 'ReciprocalSqrtKey']:
        """
        Generate two ReciprocalSqrtKey instances and initialize their attributes.

        Call the function _create_neg_exp2_table and _create_rsqrt_table to generate the negative
        exponential lookup table and the reciprocal square root lookup table. After creating two ReciprocalSqrtKey
        instances k0 and k1, a lookup key with negative exponents and reciprocal square roots is generated for each
        instance, as well as a SIGMA key.

        Args:
            num_of_keys: The number of keys.
            in_scale_bit: Scale of input bits.
            out_scale_bit: Scale of input and output bits.

        Returns:
            ReciprocalSqrtKey key pair.

        Examples:
            >>> k0, k1 = ReciprocalSqrtKey.gen(10)
        """
        down_bound = -SCALE_BIT
        upper_bound = SCALE_BIT + 1

        neg_exp2_table = _create_neg_exp2_table(down_bound, upper_bound)
        rec_sqrt_table = _create_rsqrt_table(in_scale_bit, out_scale_bit)
        k0, k1 = ReciprocalSqrtKey(), ReciprocalSqrtKey()
        k0.neg_exp2_table = k1.neg_exp2_table = neg_exp2_table
        k0.rec_sqrt_table = k1.rec_sqrt_table = rec_sqrt_table

        k0.neg_exp2_look_up_key, k1.neg_exp2_look_up_key = LookUpKey.gen(num_of_keys, down_bound, upper_bound)
        k0.rec_sqrt_look_up_key, k1.rec_sqrt_look_up_key = LookUpKey.gen(num_of_keys, 0, 8191)
        k0.sigma_key, k1.sigma_key = SigmaDICFKey.gen(num_of_keys)

        return k0, k1


def _create_rsqrt_table(in_scale_bit: int = SCALE_BIT, out_scale_bit: int = SCALE_BIT) -> RingTensor:
    """
    Create a reciprocal square root lookup table.

    Generates a square root reciprocal table based on the calculated q and converts it to the specified data type(RingTensor).

    Args:
        in_scale_bit: The scale of the input.
        out_scale_bit: The scale of the output.

    Returns:
        Square root reciprocal table.
    """
    i = torch.arange(0, 2 ** 13 - 1, dtype=torch.float64)
    e = i % 64
    m = i // 64
    q = 2 ** e * (1 + m / 128)

    rec_sqrt_table = torch.sqrt(2 ** in_scale_bit / q) * 2 ** out_scale_bit
    rec_sqrt_table = rec_sqrt_table.to(data_type)
    rec_sqrt_table = RingTensor(rec_sqrt_table, 'float')

    return rec_sqrt_table


def _create_neg_exp2_table(down_bound: int, upper_bound: int) -> RingTensor:
    """
    Create a negative exponential lookup table.

    First, creates a tensor that ranges from down_bound to upper_bound. Then use RingTensor.exp2 to compute and generate a lookup table with negative index values.

    Args:
        down_bound: Lower bound.
        upper_bound: Upper bound.

    Returns:
        Negative index lookup table.
    """
    i = RingTensor.arange(down_bound, upper_bound)
    table = RingTensor.exp2(-i)
    return table
