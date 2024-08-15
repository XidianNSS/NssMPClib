import torch

from NssMPC.config import SCALE_BIT, GELU_TABLE_BIT
from NssMPC.crypto.aux_parameter.look_up_table_keys.gelu_key import GeLUKey
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional import b2a

from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing

from NssMPC.crypto.protocols.look_up_table import LookUp
from NssMPC.crypto.primitives.function_secret_sharing.dicf import SigmaDICF
from NssMPC.crypto.protocols.selection.selectlin import SelectLin


def _gelu_forward_cpu(x):
    """
    This function is used to calculate the value of GeLU(x) on the CPU
    Args:
        x:

    Returns: the value of GeLU(x)

    """
    table_scale_bit = GELU_TABLE_BIT
    table_size = 2 ** (table_scale_bit + 2)
    y = x / (x.scale // (2 ** table_scale_bit))
    key = x.party.get_param(GeLUKey, x.numel())

    d = y >= 0
    p = d * y
    a = p * 2 - y  # abs(y)

    i = (a - RingTensor.convert_to_ring(table_size)) < 0
    c = i * a + RingTensor.convert_to_ring(table_size - 1)
    c.dtype = 'int'
    return d * x - LookUp.eval(c, key.look_up_key, key.look_up_table)


def _gelu_select_eval(x_shift: RingTensor, s_shift, key, r_in_1, r_in_2, party):
    shape = x_shift.shape
    x_shift = x_shift.flatten()
    return ArithmeticSecretSharing(RingTensor.where(s_shift, (party.party_id - r_in_1) * x_shift - r_in_2 + key.w
                                                    , r_in_1 * x_shift + key.w - key.z).reshape(shape), party)


def _gelu_forward_gpu(x):
    """
    This function is used to calculate the value of GeLU(x) on the GPU
    Args:
        x:

    Returns: the value of GeLU(x)
    """
    table_scale_bit = GELU_TABLE_BIT
    shape = x.shape
    x = x.flatten()

    gelu_key = x.party.get_param(GeLUKey, x.numel())
    sigma_key = gelu_key.sigma_key
    select_lin_key = gelu_key.select_lin_key
    select_key = gelu_key.select_key

    x_r_in = gelu_key.sigma_key.r_in
    x_shift = ArithmeticSecretSharing(x_r_in, x.party) + x.flatten()
    x_shift = x_shift.restore()

    y_shift = x_shift // (x.scale // (2 ** table_scale_bit))
    y_shift.bit_len = x.bit_len - SCALE_BIT + table_scale_bit

    d_and_w = SigmaDICF.one_key_eval(
        [y_shift, y_shift + (2 ** (table_scale_bit + 2) - 1), y_shift - (2 ** (table_scale_bit + 2))], sigma_key,
        x.party.party_id)
    d = d_and_w[0]
    w = d_and_w[1] ^ d_and_w[2]

    d_and_w_b = RingTensor.cat([d, w], dim=0)
    d_and_w_a = b2a(d_and_w_b, x.party)
    d = d_and_w_a[:d.numel()]
    w = d_and_w_a[d.numel():]

    w_shift = ArithmeticSecretSharing(select_lin_key.w, w.party) + w.flatten()
    d_shift = ArithmeticSecretSharing(select_lin_key.d, d.party) + d.flatten()

    length = w_shift.numel()
    w_and_d = ArithmeticSecretSharing.cat([w_shift, d_shift], dim=0).restore()
    w_shift = w_and_d[:length]
    d_shift = w_and_d[length:]

    c = SelectLin.eval(y_shift, w_shift, d_shift, select_lin_key)
    c.party = x.party

    s_shift = d_shift % 2
    s_shift.bit_len = d_shift.bit_len
    relu_x = _gelu_select_eval(x_shift, s_shift, select_key, select_lin_key.d, x_r_in, x.party)
    relu_x.dtype = x.dtype

    return (relu_x - LookUp.eval(c, gelu_key.look_up_key, gelu_key.look_up_table)).reshape(shape)


class SecReLU(torch.nn.Module):
    def __init__(self, inplace=True):
        super(SecReLU, self).__init__()

    def forward(self, x):
        return (x > 0) * x


def _SecReLU(x):
    return SecReLU()(x)


class SecGELU(torch.nn.Module):
    def __init__(self, approximate='none'):
        super(SecGELU, self).__init__()

    def forward(self, x):
        assert isinstance(x, ArithmeticSecretSharing), f"unsupported data type(s) for GeLU: {type(x)}"
        if x.device == 'cpu':
            return _gelu_forward_cpu(x)
        else:
            return _gelu_forward_gpu(x)


def _SecGELU(x):
    return SecGELU()(x)


class SecSoftmax(torch.nn.Module):
    def __init__(self, dim=-1):
        super(SecSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        max_x = x.__class__.max(x, dim=self.dim)
        delta_x = x - max_x
        neg_exp_x = x.__class__.exp(delta_x)
        sum_neg_exp_x = neg_exp_x.sum(dim=self.dim).unsqueeze(self.dim)

        return neg_exp_x / sum_neg_exp_x


def _SecSoftmax(x):
    return SecSoftmax(-1)(x)


class SecTanh(torch.nn.Module):
    def __init__(self, inplace=True):
        super(SecTanh, self).__init__()

    def forward(self, x):
        return x.__class__.tanh(x)


def _SecTanh(x):
    return SecTanh()(x)
