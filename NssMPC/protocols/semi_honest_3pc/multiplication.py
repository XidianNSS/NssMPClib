#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
from NssMPC.infra.mpc.party import Party, PartyCtx
from NssMPC.infra.tensor import RingTensor
from NssMPC.primitives.secret_sharing import ReplicatedSecretSharing
from NssMPC.protocols.semi_honest_3pc.truncate import truncate


def mul(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing, party: Party = None) -> ReplicatedSecretSharing:
    """Perform element-wise multiplication of RSS shares with truncation.

    This function computes the product z = x * y and applies truncation to adjust the scale.

    Args:
        x: The first operand (multiplicand).
        y: The second operand (multiplier).
        party: The party instance managing the communication. Defaults to None.

    Returns:
        The result of multiplication with adjusted scale.

    Examples:
        >>> z = mul(x, y)
    """
    if party is None:
        party = PartyCtx.get()
    z_shared = RingTensor.mul(x.item[0], y.item[0]) + RingTensor.mul(x.item[0], y.item[1]) + RingTensor.mul(x.item[1],
                                                                                                            y.item[0])
    result = ReplicatedSecretSharing.reshare(z_shared, party)
    if result.dtype == "float":
        result = truncate(result, party=party)
    return result


def matmul(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing, party: Party = None) -> ReplicatedSecretSharing:
    """Perform matrix multiplication of RSS shares with truncation.

    This function computes the matrix product Z = X @ Y and applies truncation to adjust the scale.

    Args:
        x: The left input matrix.
        y: The right input matrix.
        party: The party instance managing the communication. Defaults to None.

    Returns:
        The result of matrix multiplication with adjusted scale.

    Examples:
        >>> z = matmul(x, y)
    """
    if party is None:
        party = PartyCtx.get()
    t_i = \
        RingTensor.matmul(x.item[0], y.item[0]) + \
        RingTensor.matmul(x.item[0], y.item[1]) + \
        RingTensor.matmul(x.item[1], y.item[0])

    result = ReplicatedSecretSharing.reshare(t_i, party)
    if result.dtype == "float":
        result = truncate(result, party=party)
    return result


def mul_with_out_trunc(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing,
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
        >>> z = mul_with_out_trunc(x, y)
    """
    if party is None:
        party = PartyCtx.get()
    z_shared = RingTensor.mul(x.item[0], y.item[0]) + RingTensor.mul(x.item[0], y.item[1]) + RingTensor.mul(x.item[1],
                                                                                                            y.item[0])
    result = ReplicatedSecretSharing.reshare(z_shared, party)
    return result


def matmul_with_out_trunc(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing,
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
        >>> z = matmul_with_out_trunc(x, y)
    """
    if party is None:
        party = PartyCtx.get()
    t_i = \
        RingTensor.matmul(x.item[0], y.item[0]) + \
        RingTensor.matmul(x.item[0], y.item[1]) + \
        RingTensor.matmul(x.item[1], y.item[0])

    result = ReplicatedSecretSharing.reshare(t_i, party)
    return result
