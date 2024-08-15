from NssMPC.config import DEBUG_LEVEL, HALF_RING
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.vsigma_key import VSigmaKey
from NssMPC.crypto.primitives.function_secret_sharing.vdpf import VDPF


class VSigma(object):
    @staticmethod
    def gen(num_of_keys):
        return VSigmaKey.gen(num_of_keys)

    @staticmethod
    def eval(x_shift, keys, party_id):
        return verifiable_sigma_eval(party_id, keys, x_shift)

    @staticmethod
    def cmp_eval(x, keys, party_id):
        from NssMPC import ArithmeticSecretSharing
        if DEBUG_LEVEL == 2:
            x_shift = ArithmeticSecretSharing(keys.r_in, x.party) + x
        else:
            x_shift = ArithmeticSecretSharing(keys.r_in.reshape(x.shape), x.party) + x
        x_shift = x_shift.restore()
        return verifiable_sigma_eval(party_id, keys, x_shift)


def verifiable_sigma_eval(party_id, key, x_shift):
    shape = x_shift.shape
    x_shift = x_shift.reshape(-1, 1)
    K, c = key
    y = x_shift % (HALF_RING - 1)
    y = y + 1
    out, pi = VDPF.ppq(y, K, party_id)
    out = x_shift.signbit() * party_id ^ c.reshape(-1, 1) ^ out
    return out.reshape(shape), pi
