from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional import b2a
from NssMPC.crypto.primitives.function_secret_sharing.dicf import SigmaDICF
from NssMPC.config import SCALE_BIT
from NssMPC.crypto.aux_parameter.look_up_table_keys.div_key import DivKey
from NssMPC.crypto.protocols.look_up_table import LookUp


def secure_div(x, y):
    """
    Implement ASS division protocols using iterative method
    Correct only if y > 0 and y < 2 ** 2f
    TODO： support y in other range

    Args:
       x: dividend
       y: divisor

    Returns:
       quotient
    """
    div_key = x.party.get_param(DivKey, x.numel())
    sigma_key = div_key.sigma_key
    nexp2_key = div_key.neg_exp2_key

    y_shape = y.shape
    y_shift = x.__class__(sigma_key.r_in, x.party) + y.flatten()
    y_shift = y_shift.restore()
    y_shift = y_shift.view(y_shape)

    y_minus_powers = [y_shift - (2 ** i) for i in range(1, 2 * SCALE_BIT + 1)]
    k = SigmaDICF.one_key_eval(y_minus_powers, sigma_key, x.party.party_id)
    k = b2a(k, x.party).sum(dim=0)
    neg_exp2_k = LookUp.eval(k + 1, nexp2_key.look_up_key, nexp2_key.table)

    a = x * neg_exp2_k
    b = y * neg_exp2_k

    w = b * (-2) + 2.9142
    e0 = -(b * w) + 1
    e1 = e0 * e0

    return a * w * (e0 + 1) * (e1 + 1)
