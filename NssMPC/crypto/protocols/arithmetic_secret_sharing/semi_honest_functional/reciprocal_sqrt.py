from NssMPC.config import SCALE, SCALE_BIT, DTYPE
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional import b2a
from NssMPC.crypto.primitives.function_secret_sharing.dicf import SigmaDICF
from NssMPC.crypto.aux_parameter import ReciprocalSqrtKey
from NssMPC.crypto.protocols.look_up_table import LookUp


def secure_reciprocal_sqrt(x):
    """
    Correct only if x > 0 and x < 2 ** 2f
    TODO： support x in other range
    Args:
        x:

    Returns:

    """
    key = x.party.get_param(ReciprocalSqrtKey, x.numel())
    return reciprocal_sqrt_eval(x, key)


def reciprocal_sqrt_eval(x, key):
    neg_exp2_table = key.neg_exp2_table
    rec_sqrt_table = key.rec_sqrt_table
    neg_exp2_look_up_key = key.neg_exp2_look_up_key
    rec_sqrt_lut_key = key.rec_sqrt_look_up_key
    sigma_key = key.sigma_key

    x_shape = x.shape
    x_shift = x.__class__(sigma_key.r_in, x.party) + x.flatten()
    x_shift = x_shift.restore()
    x_shift = x_shift.view(x_shape)

    x_minus_powers = [x_shift - (2 ** i) for i in range(1, 2 * SCALE_BIT + 1)]
    k = SigmaDICF.one_key_eval(x_minus_powers, sigma_key, x.party.party_id)
    k = b2a(k, x.party).sum(dim=0)
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
