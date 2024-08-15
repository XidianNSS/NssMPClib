import torch

from NssMPC import RingTensor
from NssMPC.config import HALF_RING, DEVICE
from NssMPC.crypto.aux_parameter import Parameter
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.vdpf_key import VDPFKey
from NssMPC.crypto.primitives.function_secret_sharing.vdpf import VDPF


class VSigmaKey(Parameter):
    def __init__(self, ver_dpf_key=VDPFKey()):
        self.ver_dpf_key = ver_dpf_key
        self.c = None
        self.r_in = None

    def __iter__(self):
        return iter([self.r_in, self.ver_dpf_key, self.c])

    @staticmethod
    def gen(num_of_keys):
        return verifiable_sigma_gen(num_of_keys)


# verifiable MSB protocol from sigma protocol without r_out
def verifiable_sigma_gen(num_of_keys):
    r_in = RingTensor.random([num_of_keys])
    x1 = r_in
    y1 = r_in % (HALF_RING - 1)
    k0, k1 = VDPF.gen(num_of_keys, y1, RingTensor.convert_to_ring(1))
    c = x1.signbit() ^ 1
    c0 = torch.randint(0, 1, [num_of_keys], device=DEVICE)
    c0 = RingTensor.convert_to_ring(c0)
    c1 = c ^ c0

    k0 = VSigmaKey(k0)
    k1 = VSigmaKey(k1)

    k0.c = c0
    k1.c = c1

    from NssMPC import ArithmeticSecretSharing
    k0.r_in, k1.r_in = ArithmeticSecretSharing.share(r_in, 2)

    return k0, k1
