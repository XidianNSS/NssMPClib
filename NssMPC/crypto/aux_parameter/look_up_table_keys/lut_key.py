from NssMPC import RingTensor
from NssMPC.config.runtime import ParameterRegistry
from NssMPC.crypto.aux_parameter import Parameter


@ParameterRegistry.ignore()
class LookUpKey(Parameter):
    def __init__(self):
        self.onehot_value = None
        self.down_bound = None
        self.upper_bound = None
        self.phi = None

    @staticmethod
    def gen(num_of_keys, down, upper):
        upper = upper - down
        down = 0

        phi = RingTensor.random([num_of_keys], down_bound=0, upper_bound=upper)

        k0 = LookUpKey()
        k1 = LookUpKey()

        onehot_value = RingTensor.onehot(phi, num_classes=upper)
        from NssMPC.crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing

        k0.onehot_value, k1.onehot_value = ArithmeticSecretSharing.share(onehot_value, 2)

        k0.phi, k1.phi = ArithmeticSecretSharing.share(phi, 2)
        k0.down_bound = k1.down_bound = down
        k0.upper_bound = k1.upper_bound = upper

        return k0, k1
