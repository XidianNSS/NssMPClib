#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
from typing import Tuple

from NssMPC.config import SCALE_BIT
from NssMPC.infra.mpc.aux_parameter.parameter import Parameter, ParameterRegistry
from NssMPC.infra.mpc.party import Party, PartyCtx
from NssMPC.infra.tensor import RingTensor
from NssMPC.primitives.secret_sharing import AdditiveSecretSharing
from NssMPC.primitives.secret_sharing.function import SigmaDICF, SigmaDICFKey
from NssMPC.protocols.semi_honest_2pc.b2a import b2a
from NssMPC.protocols.semi_honest_2pc.look_up_table import LookUp
from NssMPC.protocols.semi_honest_2pc.look_up_table import LookUpKey


def secure_div(x: AdditiveSecretSharing, y: AdditiveSecretSharing,
               party: Party = None) -> AdditiveSecretSharing:
    """Securely computes the division of two Additive Secret Sharings using Goldschmidt's iterative method.

    This function implements a secure division protocol based on the **Goldschmidt approximation algorithm**.
    It iteratively converges to the quotient :math:`x / y` while maintaining data privacy.

    Note:
        **Constraint:** The divisor ``y`` must strictly satisfy the range :math:`0 < y < 2^{2f}`
        (where :math:`f` denotes the fixed-point scale bits).
        Support for a wider range of ``y`` is planned for future versions.

    Args:
        x: The dividend (numerator).
        y: The divisor (denominator).
        party: The party instance managing the communication. Defaults to None.

    Returns:
        The result of the division.

    Examples:
        >>> res = secure_div(x, y)
    """
    if x.numel() > y.numel():
        return x * secure_inv(y)
    else:
        if party is None:
            party = PartyCtx.get()
        neg_exp2_k = get_neg_exp2_k(y, party)
        a = x * neg_exp2_k
        b = y * neg_exp2_k
        w = b * (-2) + 2.9142
        e0 = -(b * w) + 1
        e1 = e0 * e0

        return a * w * (e0 + 1) * (e1 + 1)


def secure_inv(x: AdditiveSecretSharing) -> AdditiveSecretSharing:
    """Securely computes the multiplicative inverse (reciprocal) :math:`1/x`.

    This function implements the **Newton-Raphson method** optimized for SMPC.
    It normalizes the input ``x`` to the interval :math:`[0.5, 1)` using bit-level manipulation,
    computes an initial approximation using linear fitting (:math:`w \approx -2b + 2.9142`),
    and refines the result using a polynomial expansion to minimize communication rounds.

    The refinement logic corresponds to:
    :math:`result = w \cdot (1 + \epsilon) \cdot (1 + \epsilon^2)`
    where :math:`\epsilon = 1 - b \cdot w`.

    Args:
        x: The input secret sharing value to be inverted. It should be non-zero.

    Returns:
        The approximate multiplicative inverse of ``x``.

    Examples:
        >>> inv_x = secure_inv(x)
    """
    neg_exp2_k = get_neg_exp2_k(x)
    b = x * neg_exp2_k
    w = b * (-2) + 2.9142
    e0 = -(b * w) + 1
    e1 = e0 * e0

    return w * neg_exp2_k * (e0 + 1) * (e1 + 1)


def get_neg_exp2_k(divisor: AdditiveSecretSharing, party: Party = None) -> RingTensor:
    """Computes the negative power of 2 for the normalization factor k.

    Used in the division protocol to normalize the divisor.

    Args:
        divisor: The divisor to be normalized.
        party: The party instance. Defaults to None.

    Returns:
        The value :math:`2^{-k}` used for normalization.

    Examples:
        >>> k_val = get_neg_exp2_k(y)
    """
    if party is None:
        party = PartyCtx.get()
    div_key = party.get_param(DivKey, divisor.numel())
    sigma_key = div_key.sigma_key
    nexp2_key = div_key.neg_exp2_key

    y_shape = divisor.shape
    y_shift = divisor.__class__(sigma_key.r_in) + divisor.flatten()
    y_shift = y_shift.restore()
    y_shift = y_shift.view(y_shape)

    y_minus_powers = [y_shift - (2 ** i) for i in range(1, 2 * SCALE_BIT + 1)]
    k = SigmaDICF.one_key_eval(y_minus_powers, sigma_key, party.party_id)
    k = b2a(k, party).sum(dim=0)
    return LookUp.eval(k + 1, nexp2_key.look_up_key, nexp2_key.table)


class DivKey(Parameter):
    """The DivKey class for generating and managing division keys.

    Attributes:
        neg_exp2_key (_NegExp2Key): Used to store keys associated with negative exponents.
        sigma_key (SigmaDICFKey): The Sigma protocol generates the key of DICF.
    """

    @ParameterRegistry.ignore()
    class _NegExp2Key(Parameter):
        """Used to store keys associated with negative exponents.

        Attributes:
            look_up_key (LookUpKey): The key that stores the lookup table.
            table (RingTensor): The lookup table.
        """

        def __init__(self):
            """
            ATTRIBUTES:
                * **look_up_table** (*LookUpKey*): The key that stores the lookup table
                * **table** (*RingTensor*): The lookup table
            """
            self.look_up_key = LookUpKey()
            self.table = None

        def __getitem__(self, item: int) -> 'DivKey._NegExp2Key':
            """Overrides index access to return the key and assign the table attribute to it.

            This allows instances using NegExp2Key to access elements like a dictionary and automatically associate corresponding lookup tables.

            Args:
                item: Index to find.

            Returns:
                Key corresponding to the index.

            Examples:
                >>> key_item = neg_exp2_key[0]
            """
            key = super(DivKey._NegExp2Key, self).__getitem__(item)
            key.table = self.table
            return key

    def __init__(self):
        """
        Attribute:
            * **neg_exp2_key** (*_NegExp2Key*): Used to store keys associated with negative exponents.
            * **sigma_key** (*SigmaDICFKey*): The Sigma protocol generates the key of DICF
        """
        self.neg_exp2_key = DivKey._NegExp2Key()
        self.sigma_key = SigmaDICFKey()

    def __len__(self) -> int:
        """Returns the length of sigma_key.

        Returns:
            The length of sigma_key, which means that sigma_key should be an object whose length can be calculated.

        Examples:
            >>> length = len(div_key)
        """
        return len(self.sigma_key)

    @staticmethod
    def _gen_neg_exp2_key(num_of_keys: int) -> Tuple[_NegExp2Key, _NegExp2Key]:
        """Generates a negative exponential lookup table based on the given lower and upper limits.

        First set the upper and lower bounds of the lookup table, and then create two pairs of _NegExp2Key instances
        k0 and k1. Then use :meth:`~NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key.LookUpKey.gen` to create
        lookup keys associated with negative exponens and assign the results to ``k0.look_up_key`` and
        ``k1.look_up_key``, respectively. Finally, :func:`_create_neg_exp2_table` is called to generate a negative
        exponential lookup table assigned to ``k0.table`` and ``k1.table`` respectively.

        Args:
            num_of_keys: The number of keys to be generated.

        Returns:
            _NegExp2Key key pairs k0 and k1.

        Examples:
            >>> k0, k1 = DivKey._gen_neg_exp2_key(10)
        """
        down_bound = -SCALE_BIT
        upper_bound = SCALE_BIT + 1

        k0, k1 = DivKey._NegExp2Key(), DivKey._NegExp2Key()

        k0.look_up_key, k1.look_up_key = LookUpKey.gen(num_of_keys, down_bound, upper_bound)
        k0.table = k1.table = _create_neg_exp2_table(down_bound, upper_bound)

        return k0, k1

    @staticmethod
    def gen(num_of_keys: int) -> Tuple['DivKey', 'DivKey']:
        """Generates a pair of DivKey instances and initializes the associated negative exponential and sigma keys for each of them.

        First create two DivKey instances ``k0`` and ``k1``, then call the :meth:`_gen_neg_exp2_key` method to
        generate a pair of negative exponential lookup keys and assign the results to ``k0.neg_exp2_key`` and
        ``k1.neg_exp2_key``, respectively. The
        :meth:`~NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dicf_key.SigmaDICFKey.gen` method is then
        called to generate a SigmaDICFKey pair and assign the result to ``k0.sigma_key`` and ``k1.sigma_key``,
        respectively.

        Args:
            num_of_keys: The number of keys to be generated.

        Returns:
            DivKey key pairs k0 and k1.

        Examples:
            >>> k0, k1 = DivKey.gen(10)
        """
        k0, k1 = DivKey(), DivKey()
        k0.neg_exp2_key, k1.neg_exp2_key = DivKey._gen_neg_exp2_key(num_of_keys)
        k0.sigma_key, k1.sigma_key = SigmaDICFKey.gen(num_of_keys)

        return k0, k1


def _create_neg_exp2_table(down_bound: int, upper_bound: int) -> RingTensor:
    """Generates a negative exponential lookup table.

    First create a tensor from *down_bound* to *upper_bound*, then use the method
    :meth:`~NssMPC.infra.ring.ring_tensor.RingTensor.exp2` to compute the quadratic exponent of **-i** and return
    the result.

    Args:
        down_bound: Represents the lower bound of the lookup table.
        upper_bound: Represents the upper bound of the lookup table.

    Returns:
        The negative exponent of each integer from down_bound to upper_bound.

    Examples:
        >>> table = _create_neg_exp2_table(-10, 10)
    """
    i = RingTensor.arange(down_bound, upper_bound)
    table = RingTensor.exp2(-i)
    return table
