#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
from nssmpc.infra.mpc.party import Party, PartyCtx
from nssmpc.infra.tensor import RingTensor
from nssmpc.primitives.secret_sharing import ReplicatedSecretSharing
from nssmpc.protocols.honest_majority_3pc.truncation import hm3pc_truncate_aby3


def sh3pc_mul(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing, party: Party = None) -> ReplicatedSecretSharing:
    """Perform element-wise multiplication of RSS shares with truncation.

    This function computes the product z = x * y and applies truncation to adjust the scale.

    Args:
        x: The first operand (multiplicand).
        y: The second operand (multiplier).
        party: The party instance managing the communication. Defaults to None.

    Returns:
        The result of multiplication with adjusted scale.

    Examples:
        >>> z = sh3pc_mul(x, y)
    """
    if party is None:
        party = PartyCtx.get()
    z_shared = RingTensor.mul(x.item[0], y.item[0]) + RingTensor.mul(x.item[0], y.item[1]) + RingTensor.mul(x.item[1],
                                                                                                            y.item[0])
    result = ReplicatedSecretSharing.reshare(z_shared, party)
    if result.dtype == "float":
        result = hm3pc_truncate_aby3(result, party=party)
    return result


def sh3pc_matmul(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing,
                 party: Party = None) -> ReplicatedSecretSharing:
    """Perform matrix multiplication of RSS shares with truncation.

    This function computes the matrix product Z = X @ Y and applies truncation to adjust the scale.

    Args:
        x: The left input matrix.
        y: The right input matrix.
        party: The party instance managing the communication. Defaults to None.

    Returns:
        The result of matrix multiplication with adjusted scale.

    Examples:
        >>> z = sh3pc_matmul(x, y)
    """
    if party is None:
        party = PartyCtx.get()
    t_i = \
        RingTensor.matmul(x.item[0], y.item[0]) + \
        RingTensor.matmul(x.item[0], y.item[1]) + \
        RingTensor.matmul(x.item[1], y.item[0])

    result = ReplicatedSecretSharing.reshare(t_i, party)
    if result.dtype == "float":
        result = hm3pc_truncate_aby3(result, party=party)
    return result


def sh3pc_mul_with_out_trunc(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing,
                             party: Party = None) -> ReplicatedSecretSharing:
    """Perform element-wise multiplication of RSS shares without truncation.

    This function computes the product z = x * y.

    Warning:
        Since no truncation is performed, the scale of the result will be double that of the inputs
        (e.g., if inputs have scale f, the result has scale 2f).
        This is typically an intermediate step before a truncation protocol.

    Args:
        x: The first operand (multiplicand).
        y: The second operand (multiplier).
        party: The party instance managing the communication. Defaults to None.

    Returns:
        The result of multiplication with expanded scale.

    Examples:
        >>> z = sh3pc_mul_with_out_trunc(x, y)
    """
    if party is None:
        party = PartyCtx.get()
    z_shared = RingTensor.mul(x.item[0], y.item[0]) + RingTensor.mul(x.item[0], y.item[1]) + RingTensor.mul(x.item[1],
                                                                                                            y.item[0])
    result = ReplicatedSecretSharing.reshare(z_shared, party)
    return result


def sh3pc_matmul_with_out_trunc(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing,
                                party: Party = None) -> ReplicatedSecretSharing:
    """Perform matrix multiplication of RSS shares without truncation.

    This function computes the matrix product Z = X @ Y.
    Like the element-wise version, this operation does not truncate the decimal part, resulting in a doubled scale factor.

    Args:
        x: The left input matrix.
        y: The right input matrix.
        party: The party instance managing the communication. Defaults to None.

    Returns:
        The result of matrix multiplication with expanded scale.

    Examples:
        >>> z = sh3pc_matmul_with_out_trunc(x, y)
    """
    if party is None:
        party = PartyCtx.get()
    t_i = \
        RingTensor.matmul(x.item[0], y.item[0]) + \
        RingTensor.matmul(x.item[0], y.item[1]) + \
        RingTensor.matmul(x.item[1], y.item[0])

    result = ReplicatedSecretSharing.reshare(t_i, party)
    return result
