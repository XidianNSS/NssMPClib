from NssMPC import RingTensor
from NssMPC.config.runtime import ParameterRegistry
from NssMPC.crypto.aux_parameter import Parameter


@ParameterRegistry.ignore()
class SelectLinKey(Parameter):
    def __init__(self):
        self.p = None
        self.q = None
        self.w = None
        self.d = None

    @staticmethod
    def gen(num_of_keys, p, q):
        w = RingTensor.random([num_of_keys], down_bound=0, upper_bound=2)
        d = RingTensor.random([num_of_keys], down_bound=0, upper_bound=2)
        i = (w * 2 + d) % 4

        p = RingTensor.rotate(p, i.tolist())
        q = RingTensor.rotate(q, i.tolist())

        k0 = SelectLinKey()
        k1 = SelectLinKey()

        from NssMPC import ArithmeticSecretSharing
        k0.p, k1.p = ArithmeticSecretSharing.share(p, 2)
        k0.q, k1.q = ArithmeticSecretSharing.share(q, 2)

        k0.w, k1.w = ArithmeticSecretSharing.share(w, 2)
        k0.d, k1.d = ArithmeticSecretSharing.share(d, 2)

        k0.w = k0.w.item
        k1.w = k1.w.item
        k0.d = k0.d.item
        k1.d = k1.d.item

        return k0, k1
