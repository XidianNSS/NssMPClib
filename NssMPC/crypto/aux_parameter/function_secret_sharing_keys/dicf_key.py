from NssMPC import RingTensor
from NssMPC.config import HALF_RING, DEVICE, BIT_LEN
from NssMPC.crypto.aux_parameter import Parameter
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dcf_key import DCFKey
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dpf_key import DPFKey


class DICFKey(Parameter):
    """
    The secret sharing key for distributed interval comparison function

    The generation of the DICF key is based on the DCF key

    Attributes:
        dcf_key: the key of DICF
        r_in: the offset of function
        z: the check bit
    """

    def __init__(self):
        self.dcf_key = DCFKey()
        self.r_in = None
        self.z = None
        self.size = 0

    @staticmethod
    def gen(num_of_keys, down_bound=RingTensor(0), upper_bound=RingTensor(HALF_RING - 1)):
        """
        The api generating DICF key, which can generate multiple keys.

        Args:
            num_of_keys: the number of DICF key
            down_bound: the down bound of the DICF, default is 0
            upper_bound: the upper bound of the DICF, default is HALF_RING-1

        Returns:
            the participants' keyes
        """
        upper_bound = upper_bound.tensor
        down_bound = down_bound.tensor

        r_in = RingTensor.random([num_of_keys, 1], dtype='int')
        gamma = r_in - 1

        r_tensor = r_in.tensor
        # 修正参数
        q1 = (upper_bound + 1)
        ap = (down_bound + r_tensor)
        aq = (upper_bound + r_tensor)
        aq1 = (upper_bound + 1 + r_tensor)

        out = ((ap > aq) + 0) - ((ap > down_bound) + 0) + ((aq1 > q1) + 0) + ((aq == -1) + 0)

        k0 = DICFKey()
        k1 = DICFKey()

        keys = DCFKey.gen(num_of_keys, gamma, RingTensor(1))

        k0.dcf_key, k1.dcf_key = keys

        z_share = RingTensor.random([num_of_keys], device=DEVICE)
        r_share = RingTensor.random([num_of_keys], device=DEVICE)

        k0.z, k1.z = out.squeeze(1) - z_share, z_share
        k0.r_in, k1.r_in = r_in.squeeze(1) - r_share, r_share

        return k0, k1


class GrottoDICFKey(Parameter):
    def __init__(self):
        self.dpf_key = DPFKey()
        self.r_in = None

    @staticmethod
    def gen(num_of_keys, beta=RingTensor(1)):
        k0, k1 = GrottoDICFKey(), GrottoDICFKey()
        k0.r_in = RingTensor.random([num_of_keys])
        k1.r_in = RingTensor.random([num_of_keys])
        k0.dpf_key, k1.dpf_key = DPFKey.gen(num_of_keys, k0.r_in + k1.r_in, beta)
        return k0, k1


class SigmaDICFKey(Parameter):
    def __init__(self):
        self.dpf_key = DPFKey()
        self.c = None
        self.r_in = None
        self.size = 0

    @staticmethod
    def gen(num_of_keys, bit_len=BIT_LEN):
        k0 = SigmaDICFKey()
        k1 = SigmaDICFKey()

        k0.r_in = RingTensor.random([num_of_keys], down_bound=-2 ** (bit_len - 1), upper_bound=2 ** (bit_len - 1) - 1)
        k1.r_in = RingTensor.random([num_of_keys], down_bound=-2 ** (bit_len - 1), upper_bound=2 ** (bit_len - 1) - 1)
        r_in = k0.r_in + k1.r_in
        r_in.bit_len = bit_len

        y1 = r_in % (HALF_RING - 1)
        k0.dpf_key, k1.dpf_key = DPFKey.gen(num_of_keys, y1, RingTensor(1))
        c = r_in.signbit()
        c0 = RingTensor.random([num_of_keys], down_bound=0, upper_bound=2)
        c1 = c ^ c0

        k0.c = c0
        k1.c = c1

        return k0, k1
