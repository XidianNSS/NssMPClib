#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch

from NssMPC.config import SCALE_BIT, GELU_TABLE_BIT, data_type, DEVICE, BIT_LEN, HALF_RING
from NssMPC.infra.mpc.aux_parameter.parameter import Parameter, ParameterRegistry
from NssMPC.infra.mpc.party import Party, PartyCtx
from NssMPC.infra.tensor import RingTensor
from NssMPC.primitives.secret_sharing import AdditiveSecretSharing
from NssMPC.primitives.secret_sharing.function import SigmaDICFKey, DPFKey, SigmaDICF
from NssMPC.protocols.semi_honest_2pc.b2a import b2a
from NssMPC.protocols.semi_honest_2pc.look_up_table import LookUpKey, LookUp
from NssMPC.protocols.semi_honest_2pc.selectlin import SelectLinKey, SelectLin


def _gelu_forward_cpu(x,party:Party=None):
    """
    Calculates the value of GeLU(x) on the CPU.

    First, the input `x` is scaled to standardize the input. The `Party.get_param` method is then used to obtain GELU related parameters, and the object key containing the
    lookup table key and other parameters is obtained.

    Then the absolute value (a) is taken for the processed `y`.

    When calculating ``a``, if the absolute value of ``y`` is very large, it may cause a value that exceeds ``table_size``.When ``i``
    is False, it indicates that ``a`` has gone beyond the range of the lookup table. Setting ``c`` to ``table_size - 1`` is a
    safety measure to ensure that even if the input causes ``a`` to go beyond the range, the program will not crash or
    produce an incorrect lookup.

    Finally, the result is returned: when ``d`` is **True**, the original x is returned; when ``d`` is **False**, the value from the lookup table is looked up and subtracted from it.

    Args:
        x (AdditiveSecretSharing): The input RingTensor.
        party: The party instance. Defaults to None.

    Returns:
        AdditiveSecretSharing: The value of GeLU(x).

    Examples:
        >>> res = _gelu_forward_cpu(x)
    """
    if party is None:
        party = PartyCtx.get()
    table_scale_bit = GELU_TABLE_BIT
    table_size = 2 ** (table_scale_bit + 2)
    y = x / (x.scale // (2 ** table_scale_bit))
    key = party.get_param(GeLUKey, x.numel())

    d = y >= 0
    p = d * y
    a = p * 2 - y  # abs(y)

    i = (a - RingTensor.convert_to_ring(table_size)) < 0
    c = i * a + RingTensor.convert_to_ring(table_size - 1)
    c.dtype = 'int'
    return d * x - LookUp.eval(c, key.look_up_key, key.look_up_table)


def _gelu_select_eval(x_shift: RingTensor, s_shift, key, r_in_1, r_in_2, party):
    """
    Calculates the weighted GELU activation function.

    First, obtain the original shape of ``x_shift`` so that it can be restored in the final output, then,
    Flatten ``x_shift`` into a one-dimensional array for subsequent mathematical operations. Then calculate the
    weighting involving ``party_id`` and the random value ``r_in_x(1,2)``, choosing one of the two operations based on the Boolean
    value of ``s_shift``:
        * If ``s_shift`` is true, select the first result.
        * Otherwise, select the second result

    Finally, the one-dimensional result is reconstructed into the original shape for output.

    Args:
        x_shift: Input a tensor, which is subjected to a certain shift operation for subsequent processing.
        s_shift (Bool or RingTensor): A boolean value or tensor that determines which part of the operation to select.
        key (SelectLinKey): It contains the parameters necessary for encryption and decryption, such as weight parameters.
        r_in_1 (RingTensor): Random number imported during secret sharing.
        r_in_2 (RingTensor): Random number imported during secret sharing.
        party (Party): Entities involved in the calculation.

    Returns:
        AdditiveSecretSharing: The weighted GELU activation.

    Examples:
        >>> res = _gelu_select_eval(x_shift, s_shift, key, r1, r2, party)
    """
    shape = x_shift.shape
    x_shift = x_shift.flatten()
    return AdditiveSecretSharing(RingTensor.where(s_shift, (party.party_id - r_in_1) * x_shift - r_in_2 + key.w
                                                  , r_in_1 * x_shift + key.w - key.z).reshape(shape))


def _gelu_forward_gpu(x,party:Party=None):
    """
    Calculates the value of GeLU(x) on the GPU.

    First, GeLU's required key parameters, including ``sigma_key``, ``select_lin_key``, and ``select_key``, are obtained from ``x``'s
    participant, and then ``x`` is encrypted (offset) and scaled to obtain ``y_shift``, The values of ``d`` and ``w`` are obtained by
    method ``SigmaDICF.one_key_eval``, and the two are offset after extracted. Calculate the result ``c`` of the linear selection using
    the ``SelectLin.eval`` method, perform the nonlinear calculation with ``_gelu_select_eval`` to get ``relu_x``, ``look_up_key`` is
    used to obtain the calculated value in ``gelu_key.look_up_table``.

    Args:
        x (AdditiveSecretSharing): Input an ASS to represent the value that needs to be computed with GeLU.
        party: The party instance. Defaults to None.

    Returns:
        AdditiveSecretSharing: The Computational Results of GeLU.

    Examples:
        >>> res = _gelu_forward_gpu(x)
    """
    if party is None:
        party = PartyCtx.get()
    table_scale_bit = GELU_TABLE_BIT
    shape = x.shape
    x = x.flatten()

    gelu_key = party.get_param(GeLUKey, x.numel())
    sigma_key = gelu_key.sigma_key
    select_lin_key = gelu_key.select_lin_key
    select_key = gelu_key.select_key

    x_r_in = gelu_key.sigma_key.r_in
    x_shift = AdditiveSecretSharing(x_r_in) + x.flatten()
    x_shift = x_shift.restore()

    y_shift = x_shift // (x.scale // (2 ** table_scale_bit))
    y_shift.bit_len = x.bit_len - SCALE_BIT + table_scale_bit

    d_and_w = SigmaDICF.one_key_eval(
        [y_shift, y_shift + (2 ** (table_scale_bit + 2) - 1), y_shift - (2 ** (table_scale_bit + 2))], sigma_key,
        party.party_id)
    d = d_and_w[0]
    w = d_and_w[1] ^ d_and_w[2]

    d_and_w_b = RingTensor.cat([d, w], dim=0)
    d_and_w_a = b2a(d_and_w_b, party)
    d = d_and_w_a[:d.numel()]
    w = d_and_w_a[d.numel():]

    w_shift = AdditiveSecretSharing(select_lin_key.w) + w.flatten()
    d_shift = AdditiveSecretSharing(select_lin_key.d) + d.flatten()

    length = w_shift.numel()
    w_and_d = AdditiveSecretSharing.cat([w_shift, d_shift], dim=0).restore()
    w_shift = w_and_d[:length]
    d_shift = w_and_d[length:]

    c = SelectLin.eval(y_shift, w_shift, d_shift, select_lin_key)

    s_shift = d_shift % 2
    s_shift.bit_len = d_shift.bit_len
    relu_x = _gelu_select_eval(x_shift, s_shift, select_key, select_lin_key.d, x_r_in, party)
    relu_x.dtype = x.dtype

    return (relu_x - LookUp.eval(c, gelu_key.look_up_key, gelu_key.look_up_table)).reshape(shape)


class SecReLU(torch.nn.Module):
    """
    Basic ReLU activation function.

    The implementation of this class is mainly based on the `paper Sonic <https://maggichk.github.io/papers/sonic.pdf>`_.

    Args:
        inplace (bool): can optionally do the operation in-place. Default: ``True``

    Examples:
        >>> relu = SecReLU()
    """

    def __init__(self, inplace=True):
        super(SecReLU, self).__init__()

    def forward(self, x):
        """
        Forward pass of SecReLU.

        ==========  =======
        condition   result
        ==========  =======
        x > 0       x
        x <= 0      0
        ==========  =======

        Args:
            x (RingTensor): The input tensor.

        Returns:
            RingTensor: Result of ReLU operation.

        Examples:
            >>> y = relu(x)
        """
        return (x >= 0) * x


def _SecReLU(x):
    """
    A simple utility function for creating a SecReLU instance and immediately applying it to input x.

    The implementation of this method is mainly based on the `paper Sonic <https://maggichk.github.io/papers/sonic.pdf>`_.

    Args:
        x (RingTensor): The input tensor.

    Returns:
        AdditiveSecretSharing: Calculation result after SecReLU operation.

    Examples:
        >>> res = _SecReLU(x)
    """
    return SecReLU()(x)


class SecGELU(torch.nn.Module):
    """
    A safe GeLU activation function is implemented.

    The implementation of this class is mainly based on the `paper Sigma <https://eprint.iacr.org/2023/1269.pdf>`_.

    .. math::

        \text{GELU}(x) = x \cdot P(X \leq x) = x \cdot \Phi(x)

    Args:
        approximate (str): used to specify the approximation method for GeLU, defaulting to 'none'.

    Examples:
        >>> gelu = SecGELU()
    """

    def __init__(self, approximate='none'):
        super(SecGELU, self).__init__()

    def forward(self, x):
        """
        Forward pass of SecGELU.

        First check if x is the ArithmeticSecretSharing type, then check x's device:
            If on a CPU, call ``_gelu_forward_cpu(x)`` for processing.
            If on a GPU, call ``_gelu_forward_gpu(x)`` for processing.

        Args:
            x (RingTensor): The input tensor.

        Returns:
            RingTensor: The result of GELU operations performed on CPU or GPU vendors.

        Examples:
            >>> y = gelu(x)
        """
        assert isinstance(x, AdditiveSecretSharing), f"unsupported data type(s) for GeLU: {type(x)}"
        if x.device == 'cpu':
            return _gelu_forward_cpu(x)
        else:
            return _gelu_forward_gpu(x)


def _SecGELU(x):
    """
    A convenient utility function for creating a SecGeLU instance and immediately applying it to input x.

    The implementation of this method is mainly based on the `paper Sigma <https://eprint.iacr.org/2023/1269.pdf>`_.

    Args:
        x (RingTensor): The input tensor.

    Returns:
        AdditiveSecretSharing: Calculation result after SecGeLU operation.

    Examples:
        >>> res = _SecGELU(x)
    """
    return SecGELU()(x)


class SecSoftmax(torch.nn.Module):
    """
    A secure softmax operation has been achieved.

    The implementation of this class is mainly based on the `paper Sigma <https://eprint.iacr.org/2023/1269.pdf>`_.

    Args:
        dim (int): Specify which dimension to operate on when calculating the softmax. Default: -1.

    Examples:
        >>> softmax = SecSoftmax(dim=1)
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
        Forward pass of SecSoftmax.

        First calculate the maximum value of the specified dimension to avoid the problem of value overflow. The
        maximum value is then subtracted from the original input to calculate the negative exponent of each element.
        Finally, the sum of all negative exponents is calculated, and the dimensional consistency is maintained by
        unsqueeze.

        Args:
            x (RingTensor): The input tensor.

        Returns:
            RingTensor: Divide the negative exponent by the sum.

        Examples:
            >>> y = softmax(x)
        """
        max_x = x.__class__.max(x, dim=self.dim)
        delta_x = x - max_x
        neg_exp_x = x.__class__.exp(delta_x)
        sum_neg_exp_x = neg_exp_x.sum(dim=self.dim).unsqueeze(self.dim)

        return neg_exp_x / sum_neg_exp_x


def _SecSoftmax(x):
    """
    A convenient utility function for creating a SecSoftmax instance and immediately applying it to input x.

    The implementation of this method is mainly based on the `paper Sigma <https://eprint.iacr.org/2023/1269.pdf>`_.

    Args:
        x (RingTensor): The input tensor.

    Returns:
        SecSoftmax: A SecSoftmax object and passes the input x to its forward method.

    Examples:
        >>> res = _SecSoftmax(x)
    """
    return SecSoftmax(-1)(x)


class SecTanh(torch.nn.Module):
    """
    The tanh (hyperbolic tangent) activation function has been implemented.

    Args:
        inplace (bool): can optionally do the operation in-place. Default: ``True``

    Examples:
        >>> tanh = SecTanh()
    """

    def __init__(self, inplace=True):
        super(SecTanh, self).__init__()

    def forward(self, x):
        """
        Forward pass of SecTanh.

        Use the tanh method to calculate the hyperbolic tangent value of the input.

        Args:
            x (RingTensor): The input tensor.

        Returns:
            RingTensor: The hyperbolic tangent value of the input ``x``.

        Examples:
            >>> y = tanh(x)
        """
        return x.__class__.tanh(x)


def _SecTanh(x):
    """
    A convenient way to create a SecTanh instance and immediately apply it to input x.

    Args:
        x (RingTensor): The input tensor.

    Returns:
        SecSoftmax: Create a SecTanh object and pass the input x to its forward method. Return the result.

    Examples:
        >>> res = _SecTanh(x)
    """
    return SecTanh()(x)


class GeLUKey(Parameter):
    """
    Generate keys for the Gaussian Error Linear Unit (GeLU) activation function.

    Attributes:
        look_up_key (LookUpKey): The key of look up table.
        sigma_key (SigmaDICFKey): The Sigma protocol generates the key of DICF.
        select_lin_key (SelectLinKey): An instance of SelectLinKey, possibly for selection operations in the computation.
        select_key (GeLUKey._SelectKey): Contains two attributes: w (weight) and z (the result of GELU operation).
        look_up_table (RingTensor): A table that holds precomputed values for the GeLU function.

    Examples:
        >>> key = GeLUKey()
    """

    @ParameterRegistry.ignore()
    class _SelectKey(Parameter):
        def __init__(self):
            self.w = None
            self.z = None

    def __init__(self):
        """
        ATTRIBUTES:
            * **look_up_key** (:class:`~NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key.LookUpKey`): The key of look up table.
            * **sigma_key** (:class:`~NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dicf_key.SigmaDICFKey`): The Sigma protocol generates the key of DICF.
            * **select_lin_key** (:class:`~NssMPC.crypto.aux_parameter.select_keys.selectlin_key.SelectLinKey`): An instance of SelectLinKey, possibly for selection operations in the computation.
            * **select_key** (:class`~NssMPC.crypto.aux_parameter.look_up_table_keys.gelu_key.GeLUKey._SelectKey`): Contains two attributes: w (weight) and z (the result of GELU operation).
            * **look_up_table** (:class:`~NssMPC.infra.ring.ring_tensor.RingTensor`): A table that holds precomputed values for the GeLU function.
        """
        self.look_up_key = LookUpKey()
        self.sigma_key = SigmaDICFKey()
        self.select_lin_key = SelectLinKey()
        self.select_key = GeLUKey._SelectKey()
        self.look_up_table = None

    def __len__(self):
        """
        Get the length of the key.

        Returns:
            int: The length of sigma_key, which means that sigma_key should be an object whose length can be calculated.

        Examples:
            >>> length = len(key)
        """
        return len(self.sigma_key)

    @staticmethod
    def _gen_select_key(s_r_in, x_r_in):
        """
        Generates two selection keys based on input values ``s_r_in`` and ``x_r_in``.

        It uses secret sharing to split these values into two shares for secure computations.

        Args:
            s_r_in (RingTensor): Blinding factor for select bit s.
            x_r_in (RingTensor): Blinding factor for data x.

        Returns:
            tuple: shared key pairs k0 and k1.

        Examples:
            >>> k0, k1 = GeLUKey._gen_select_key(s_r, x_r)
        """
        w = s_r_in * x_r_in
        z = 2 * s_r_in * x_r_in

        k0, k1 = GeLUKey._SelectKey(), GeLUKey._SelectKey()

        k0.w, k1.w = AdditiveSecretSharing.share(w, 2)
        k0.z, k1.z = AdditiveSecretSharing.share(z, 2)

        k0.w, k1.w, k0.z, k1.z = k0.w.item, k1.w.item, k0.z.item, k1.z.item
        return k0, k1

    @staticmethod
    def _gen_sigma_cmp_key(num_of_keys, table_scale_bit):
        """
        Generate the key associated with the SigmaDICFKey.

        First, the sigma keys ``k0`` and ``k1`` are initialized, a random tensor within a specified range is generated using
        the ``RingTensor.random`` method, and ``k0`` and ``k1`` are shared as function secrets. Then, the range of ``input_r_in`` is
        reduced by right-shift operation, and the DPF key is generated by ``DPFKey.gen`` after comparison point ``y1`` is
        generated. Using ``RingTensor.signbit`` to compute the input signbit, and generate a random binary tensor ``c0``, based on the
        signbit ``c`` and ``c0`` XOR to generate ``c1``, ``c0`` and ``c1`` are assigned to ``k0.c`` and ``k1.c`` respectively.

        Args:
            num_of_keys (int): The number of generated keys.
            table_scale_bit (int): The number of bits used for scaling, which affects the range of small random inputs.

        Returns:
            tuple: SigmaDICFKey key pair, randomly generated confusion factor input_r_in, reduced confusion factor.

        Examples:
            >>> k0, k1, r_in, small_r = GeLUKey._gen_sigma_cmp_key(10, 6)
        """
        k0, k1 = SigmaDICFKey(), SigmaDICFKey()

        input_r_in = RingTensor.random([num_of_keys], down_bound=-HALF_RING // 2, upper_bound=HALF_RING // 2)
        k0.r_in, k1.r_in = AdditiveSecretSharing.share(input_r_in, 2)
        k0.r_in, k1.r_in = k0.r_in.item, k1.r_in.item
        small_r = input_r_in // 2 ** (SCALE_BIT - table_scale_bit)
        small_r.bit_len = BIT_LEN - SCALE_BIT + table_scale_bit

        y1 = small_r % (HALF_RING - 1)
        k0.dpf_key, k1.dpf_key = DPFKey.gen(num_of_keys, y1, RingTensor(1))
        c = input_r_in.signbit()
        c0 = RingTensor.random([num_of_keys], down_bound=0, upper_bound=2)
        c1 = c ^ c0

        k0.c = c0
        k1.c = c1

        return k0, k1, input_r_in, small_r

    @staticmethod
    def gen(num_of_keys):
        """
        Generate two GeLUKey instances k0 and k1, and initialize various associated keys and lookup tables for them.

        First initialize the scale and size of the lookup table, create two GeLUKey objects ``k0`` and ``k1``,
        use the method ``LookUpKey.gen`` to generate the lookup key, and use the function ``_create_gelu_table`` to generate the lookup
        table. Generate sigma comparison keys using ``_gen_sigma_cmp_key``. q . The q is the search range. Call the
        ``SelectLinKey.gen`` method, passing num_of_keys, p, and q, to generate the selected linear key. Finally,
        a select key pair is generated.

        .. note::
            The default scale of elements in the gelu table is 2 ** 6.

        Args:
            num_of_keys (int): the number of keys.

        Returns:
            tuple: GeLUKey key pair.

        Examples:
            >>> k0, k1 = GeLUKey.gen(10)
        """
        table_scale_bit = GELU_TABLE_BIT
        table_scale = 2 ** table_scale_bit
        table_size = 4 * table_scale
        k0 = GeLUKey()
        k1 = GeLUKey()

        k0.look_up_key, k1.look_up_key = LookUpKey.gen(num_of_keys, 0, table_size)
        k0.look_up_table = k1.look_up_table = _create_gelu_table(table_scale_bit)

        k0.sigma_key, k1.sigma_key, input_r_in, select_lin_r = GeLUKey._gen_sigma_cmp_key(num_of_keys, table_scale_bit)

        p = RingTensor([0, 0, -1, 1]).repeat(num_of_keys, 1)

        q = RingTensor([2 ** (table_scale_bit + 2) - 1, 2 ** (table_scale_bit + 2) - 1]).repeat(num_of_keys, 1)
        q = RingTensor.cat((q, select_lin_r.view(-1, 1), (-select_lin_r).view(-1, 1)), dim=1)

        k0.select_lin_key, k1.select_lin_key = SelectLinKey.gen(num_of_keys, p, q)

        r_in_1 = (k0.select_lin_key.d + k1.select_lin_key.d) % 2

        k0.select_key, k1.select_key = GeLUKey._gen_select_key(r_in_1, input_r_in)

        return k0, k1

    def __getitem__(self, item):
        """
        This method overrides the index operation so that the elements of GeLUKey can be accessed through the index.

        Set the look_up_table of the key to the look_up_table of the current instance to ensure that the correct lookup table is used during access.

        Args:
            item (int): Index to find.

        Returns:
            GeLUKey: Key corresponding to the index.

        Examples:
            >>> k = key[0]
        """
        key = super(GeLUKey, self).__getitem__(item)
        key.look_up_table = self.look_up_table
        return key


def _create_gelu_table(table_scale_bit=GELU_TABLE_BIT):
    """
    The function is used to generate a lookup table that stores the difference between the ReLU and GeLU functions.

    Use ``torch.arange`` to generate a one-dimensional tensor from 0 to 4 * table_scale. These values are then
    normalized to the range [0, 4) by dividing by table_scale. This code ensures that the input keys we create have
    the appropriate precision. After calculating the difference between ReLU and GeLU, Convert the calculated difference to the RingTensor type.

    .. note::
        * In the input domain range, ReLU(x)-GeLU(x) is non-zero on (-4, 4), and is an even function, so the input range can be determined to be [0, 4)

        * The precision of this input is determined by table_scale_bit(f), that is, the input range [0, 2 ** (f + 2))

    Args:
        table_scale_bit (int): Determines the precision and size of the lookup table. (The default value is **GELU_TABLE_BIT**)

    Returns:
        RingTensor: the table of the ReLU(x)-GeLU(x).

    Examples:
        >>> table = _create_gelu_table()
    """

    table_scale = 2 ** table_scale_bit
    table_key = torch.arange(0, 4 * table_scale, dtype=data_type, device=DEVICE) / table_scale
    relu = torch.nn.ReLU()
    gelu = torch.nn.GELU()

    table_norm_value = relu(table_key) - gelu(table_key)
    table = RingTensor.convert_to_ring(table_norm_value)
    return table
