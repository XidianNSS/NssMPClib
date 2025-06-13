#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch

from NssMPC.config import SCALE_BIT, GELU_TABLE_BIT
from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.aux_parameter.look_up_table_keys.gelu_key import GeLUKey
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional import b2a

from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.crypto.primitives.arithmetic_secret_sharing import ArithmeticSecretSharing

from NssMPC.crypto.protocols.look_up_table import LookUp
from NssMPC.crypto.primitives.function_secret_sharing.dicf import SigmaDICF
from NssMPC.crypto.protocols.selection.selectlin import SelectLin


def _gelu_forward_cpu(x):
    """
    This function is used to calculate the value of GeLU(x) on the CPU.

    First, the input *x* is scaled to standardize the input. The :meth:`~NssMPC.secure_model.mpc_party.party.
    Party.get_param` method is then used to obtain GELU related parameters, and the object key containing the
    lookup table key and other parameters is obtained.

    Then the absolute value (a) is taken for the processed *y*.

    When calculating ``a``, if the absolute value of ``y`` is very large, it may cause a value that exceeds ``table_size``.When ``i``
    is False, it indicates that ``a`` has gone beyond the range of the lookup table. Setting ``c`` to ``table_size - 1`` is a
    safety measure to ensure that even if the input causes ``a`` to go beyond the range, the program will not crash or
    produce an incorrect lookup.

    Finally, the result is returned: when ``d`` is **True**, the original x is returned; when ``d`` is **False**, the value from the lookup table is looked up and subtracted from it.

    :param x: The input RingTensor
    :type x: ArithmeticSecretSharing
    :return: the value of GeLU(x)
    :rtype: ArithmeticSecretSharing

    """
    table_scale_bit = GELU_TABLE_BIT
    table_size = 2 ** (table_scale_bit + 2)
    y = x / (x.scale // (2 ** table_scale_bit))
    key = PartyRuntime.party.get_param(GeLUKey, x.numel())

    d = y >= 0
    p = d * y
    a = p * 2 - y  # abs(y)

    i = (a - RingTensor.convert_to_ring(table_size)) < 0
    c = i * a + RingTensor.convert_to_ring(table_size - 1)
    c.dtype = 'int'
    return d * x - LookUp.eval(c, key.look_up_key, key.look_up_table)


def _gelu_select_eval(x_shift: RingTensor, s_shift, key, r_in_1, r_in_2, party):
    """
    Calculate the weighted GELU activation function.

    First, obtain the original shape of ``x_shift`` so that it can be restored in the final output, then,
    Flatten ``x_shift`` into a one-dimensional array for subsequent mathematical operations. Then calculate the
    weighting involving ``party_id`` and the random value ``r_in_x(1,2)``, choosing one of the two operations based on the Boolean
    value of ``s_shift``:
        * If ``s_shift`` is true, select the first result.
        * Otherwise, select the second result

    Finally, the one-dimensional result is reconstructed into the original shape for output.

    :param x_shift: Input a tensor, which is subjected to a certain shift operation for subsequent processing.
    :type x_shift: RingTensor
    :param s_shift: A boolean value or tensor that determines which part of the operation to select.
    :type s_shift: Bool or RingTensor
    :param key: It contains the parameters necessary for encryption and decryption, such as weight parameters.
    :type key: :class:`~NssMPC.crypto.aux_parameter.select_keys.selectlin_key.SelectLinKey`
    :param r_in_1: Random number imported during secret sharing
    :type r_in_1: RingTensor
    :param r_in_2: Random number imported during secret sharing
    :type r_in_2: RingTensor
    :param party: Entities involved in the calculation
    :type party: Party
    :return: The weighted GELU activation
    :rtype: ArithmeticSecretSharing
    """
    shape = x_shift.shape
    x_shift = x_shift.flatten()
    return ArithmeticSecretSharing(RingTensor.where(s_shift, (party.party_id - r_in_1) * x_shift - r_in_2 + key.w
                                                    , r_in_1 * x_shift + key.w - key.z).reshape(shape))


def _gelu_forward_gpu(x):
    """
    This function is used to calculate the value of GeLU(x) on the GPU.

    First, GeLU's required key parameters, including ``sigma_key``, ``select_lin_key``, and ``select_key``, are obtained from ``x``'s
    participant, and then ``x`` is encrypted (offset) and scaled to obtain ``y_shift``, The values of ``d`` and ``w`` are obtained by
    method :meth:`~NssMPC.crypto.primitives.function_secret_sharing.dicf.SigmaDICF.one_key_eval`, and the two are offset after extracted. Calculate the result ``c`` of the linear selection using
    the :meth:`~NssMPC.crypto.protocols.selection.selectlin.SelectLin.eval` method, perform the nonlinear calculation with :func:`_gelu_select_eval` to get ``relu_x``, ``look_up_key`` is
    used to obtain the calculated value in ``gelu_key.look_up_table``.

    :param x: Input an ASS to represent the value that needs to be computed with GeLU.
    :type x: ArithmeticSecretSharing
    :return: The Computational Results of GeLU
    :rtype: ArithmeticSecretSharing
    """
    table_scale_bit = GELU_TABLE_BIT
    shape = x.shape
    x = x.flatten()

    gelu_key = PartyRuntime.party.get_param(GeLUKey, x.numel())
    sigma_key = gelu_key.sigma_key
    select_lin_key = gelu_key.select_lin_key
    select_key = gelu_key.select_key

    x_r_in = gelu_key.sigma_key.r_in
    x_shift = ArithmeticSecretSharing(x_r_in) + x.flatten()
    x_shift = x_shift.restore()

    y_shift = x_shift // (x.scale // (2 ** table_scale_bit))
    y_shift.bit_len = x.bit_len - SCALE_BIT + table_scale_bit

    d_and_w = SigmaDICF.one_key_eval(
        [y_shift, y_shift + (2 ** (table_scale_bit + 2) - 1), y_shift - (2 ** (table_scale_bit + 2))], sigma_key,
        PartyRuntime.party.party_id)
    d = d_and_w[0]
    w = d_and_w[1] ^ d_and_w[2]

    d_and_w_b = RingTensor.cat([d, w], dim=0)
    d_and_w_a = b2a(d_and_w_b, PartyRuntime.party)
    d = d_and_w_a[:d.numel()]
    w = d_and_w_a[d.numel():]

    w_shift = ArithmeticSecretSharing(select_lin_key.w) + w.flatten()
    d_shift = ArithmeticSecretSharing(select_lin_key.d) + d.flatten()

    length = w_shift.numel()
    w_and_d = ArithmeticSecretSharing.cat([w_shift, d_shift], dim=0).restore()
    w_shift = w_and_d[:length]
    d_shift = w_and_d[length:]

    c = SelectLin.eval(y_shift, w_shift, d_shift, select_lin_key)

    s_shift = d_shift % 2
    s_shift.bit_len = d_shift.bit_len
    relu_x = _gelu_select_eval(x_shift, s_shift, select_key, select_lin_key.d, x_r_in, PartyRuntime.party)
    relu_x.dtype = x.dtype

    return (relu_x - LookUp.eval(c, gelu_key.look_up_key, gelu_key.look_up_table)).reshape(shape)


class SecReLU(torch.nn.Module):
    """
    * The implementation of this class is mainly based on the `paper Sonic <https://maggichk.github.io/papers/sonic.pdf>`_.

    Basic ReLU activation function.

    Call the parent class constructor ``super(SecReLU, self).__init__()`` to initialize the base module.
    """

    def __init__(self, inplace=True):
        super(SecReLU, self).__init__()

    def forward(self, x):
        """
        ==========  =======
        condition   result
        ==========  =======
        x > 0       x
        x <= 0      0
        ==========  =======

        :param x: The input tensor
        :type x: RingTensor
        :return: Result of ReLU operation
        :rtype: RingTensor
        """
        return (x > 0) * x


def _SecReLU(x):
    """
    * The implementation of this method is mainly based on the `paper Sonic <https://maggichk.github.io/papers/sonic.pdf>`_.

    A simple utility function for creating a SecReLU instance and immediately applying it to input x.

    :param x: The input tensor
    :type x: RingTensor
    :return: Calculation result after SecReLU operation
    :rtype: ArithmeticSecretSharing
    """
    return SecReLU()(x)


class SecGELU(torch.nn.Module):
    """
    * The implementation of this class is mainly based on the `paper Sigma <https://eprint.iacr.org/2023/1269.pdf>`_.

    A safe GeLU activation function is implemented.

    .. math::

        \text{GELU}(x) = x \cdot P(X \leq x) = x \cdot \Phi(x)

    .. note::
        "approximate": used to specify the approximation method for GeLU, defaulting to 'none', but not currently used in the implementation.

    Call the parent class constructor super(SecGeLU, self).__init__() to initialize the base module.
    """

    def __init__(self, approximate='none'):
        super(SecGELU, self).__init__()

    def forward(self, x):
        """
        First check if x is the ArithmeticSecretSharing type, then check x's device:
            If on a CPU, call :func:`_gelu_forward_cpu(x)` for processing.
            If on a GPU, call :func:`_gelu_forward_gpu(x)` for processing.

        :param x: The input tensor
        :type x: RingTensor
        :return: The result of GELU operations performed on CPU or GPU vendors
        :rtype: RingTensor
        """
        assert isinstance(x, ArithmeticSecretSharing), f"unsupported data type(s) for GeLU: {type(x)}"
        if x.device == 'cpu':
            return _gelu_forward_cpu(x)
        else:
            return _gelu_forward_gpu(x)


def _SecGELU(x):
    """
     * The implementation of this method is mainly based on the `paper Sigma <https://eprint.iacr.org/2023/1269.pdf>`_.

    A convenient utility function for creating a SecGeLU instance and immediately applying it to input x.

    :param x: The input tensor
    :type x: RingTensor
    :return: Calculation result after SecGeLU operation
    :rtype: ArithmeticSecretSharing
    """
    return SecGELU()(x)


class SecSoftmax(torch.nn.Module):
    """
    * The implementation of this class is mainly based on the `paper Sigma <https://eprint.iacr.org/2023/1269.pdf>`_.

    A secure softmax operation has been achieved.

    """

    def __init__(self, dim=-1):
        """
        A call to ``super(SecSoftmax, self).__init__()`` initializes the parent module.

        :param dim: Specify which dimension to operate on when calculating the softmax.
        :type dim: int
        """
        super(SecSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        First calculate the maximum value of the specified dimension to avoid the problem of value overflow. The
        maximum value is then subtracted from the original input to calculate the negative exponent of each element.
        Finally, the sum of all negative exponents is calculated, and the dimensional consistency is maintained by
        unsqueeze.

        :param x: The input tensor
        :type x: RingTensor
        :return: Divide the negative exponent by the sum.
        :rtype: RingTensor
        """
        max_x = x.__class__.max(x, dim=self.dim)
        delta_x = x - max_x
        neg_exp_x = x.__class__.exp(delta_x)
        sum_neg_exp_x = neg_exp_x.sum(dim=self.dim).unsqueeze(self.dim)

        return neg_exp_x / sum_neg_exp_x


def _SecSoftmax(x):
    """
    * The implementation of this method is mainly based on the `paper Sigma <https://eprint.iacr.org/2023/1269.pdf>`_.

    A convenient utility function for creating a SecSoftmax instance and immediately applying it to input x.

    :param x: The input tensor
    :type x: RingTensor
    :return: A SecSoftmax object and passes the input x to its forward method.
    :rtype: SecSoftmax
    """
    return SecSoftmax(-1)(x)


class SecTanh(torch.nn.Module):
    """
    The tanh (hyperbolic tangent) activation function has been implemented.
    """

    def __init__(self, inplace=True):
        """
        Initialize the parent class module to ensure that basic functionality is available.
        """
        super(SecTanh, self).__init__()

    def forward(self, x):
        """
        Use the tanh method to calculate the hyperbolic tangent value of the input.

        :param x: The input tensor
        :type x: RingTensor
        :return: The hyperbolic tangent value of the input ``x``.
        """
        return x.__class__.tanh(x)


def _SecTanh(x):
    """
    A convenient way to create a SecTanh instance and immediately apply it to input x.

    :param x: The input tensor
    :type x: RingTensor
    :return: Create a SecTanh object and pass the input x to its forward method. Return the result.
    :rtype: SecSoftmax
    """
    return SecTanh()(x)
