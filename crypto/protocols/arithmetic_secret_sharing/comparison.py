from config.base_configs import GE_TYPE
from crypto.primitives.function_secret_sharing import DPF
from crypto.protocols.arithmetic_secret_sharing.b2a import b2a
from crypto.protocols.arithmetic_secret_sharing.most_significant_bit import get_msb
from crypto.protocols.function_secret_sharing import *


def secure_eq(x, y):
    z = x - y
    shape = z.shape
    key = x.party.get_param(PPQCompareKey, x.numel())
    z_shift = x.__class__(key.r_in, x.party) + z.flatten()
    z_shift = z_shift.restore()
    z_shift.dtype = 'int'
    ge_res = DPF.eval(z_shift, key.dpf_key, x.party.party_id).view(shape)
    ge_res.dtype = x.dtype
    return x.__class__(ge_res * x.scale, x.party)


def secure_ge(x, y):
    """
    The comparison of two ArithmeticSharedRingTensor.

    There are four different ways you can choose to compare the two ArithmeticSharedRingTensor by changing the value of
    Ge_TYPE.
    """
    ge_methods = {'MSB': msb_ge, 'DICF': dicf_ge, 'PPQ': ppq_ge, 'SIGMA': sigma_ge}
    return ge_methods[GE_TYPE](x, y)


def msb_ge(x, y):
    z = x - y
    shape = z.shape
    msb = get_msb(z)
    ge_res = msb
    if x.party.party_id == 0:
        ge_res = msb ^ 1
    ge_res = b2a(ge_res, x.party)
    ge_res = ge_res.reshape(shape)
    ge_res.dtype = x.dtype
    ge_res = ge_res * x.scale
    return ge_res


def dicf_ge(x, y):
    z = x - y
    shape = z.shape
    key = x.party.get_param(DICFKey, x.numel())
    z_shift = x.__class__(key.r_in, x.party) + z.flatten()
    z_shift = z_shift.restore()
    z_shift.dtype = 'int'
    ge_res = DICF.eval(z_shift, key, x.party.party_id).view(shape) * x.scale
    ge_res.dtype = x.dtype
    return x.__class__(ge_res, x.party)


def ppq_ge(x, y):
    z = x - y
    shape = z.shape
    key = x.party.get_param(PPQCompareKey, x.numel())
    z_shift = x.__class__(key.r_in, x.party) - z.flatten()
    z_shift = z_shift.restore()
    z_shift.dtype = 'int'
    ge_res = PPQCompare.eval(z_shift, key, x.party.party_id).view(shape)
    ge_res = b2a(ge_res, x.party)
    ge_res.dtype = x.dtype
    return ge_res * x.scale


def sigma_ge(x, y):
    z = x - y
    shape = z.shape
    key = x.party.get_param(SigmaCompareKey, x.numel())
    z_shift = x.__class__(key.r_in, x.party) + z.flatten()
    z_shift = z_shift.restore()
    z_shift.dtype = 'int'
    ge_res = SigmaCompare.eval(z_shift, key, x.party.party_id).view(shape)
    ge_res = b2a(ge_res, x.party)
    ge_res.dtype = x.dtype
    return ge_res * x.scale
