from NssMPC.common.ring.ring_tensor import RingTensor

from NssMPC.crypto.aux_parameter import Parameter


class B2AKey(Parameter):
    def __init__(self):
        self.r = None

    @staticmethod
    def gen(num_of_params):
        r = RingTensor.random([num_of_params], down_bound=0, upper_bound=2)
        k0, k1 = B2AKey(), B2AKey()
        from NssMPC.crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing
        k0.r, k1.r = ArithmeticSecretSharing.share(r, 2)

        return k0, k1
