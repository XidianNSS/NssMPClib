from NssMPC import RingTensor
from NssMPC.crypto.aux_parameter import Parameter
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.vdpf_key import VDPFKey


class VOSKey(Parameter):
    def __init__(self, r00=None, k00=VDPFKey()):
        self.r00 = r00
        self.k00 = k00

    def __iter__(self):
        return iter([self.r00, self.k00])

    @staticmethod
    def gen(num_of_keys):
        r00 = RingTensor.random([num_of_keys])
        r01 = RingTensor.random([num_of_keys])
        r0 = r00 + r01
        k00, k01 = VDPFKey.gen(num_of_keys, r0, RingTensor.convert_to_ring(1))
        VOSKey.check_keys_and_r()
        return VOSKey(r00=r00, k00=k00), VOSKey(r00=r01, k00=k01)

    @staticmethod
    def check_keys_and_r():
        pass
