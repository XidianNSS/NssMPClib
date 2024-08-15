from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.config import SCALE_BIT
from NssMPC.config.runtime import ParameterRegistry
from NssMPC.crypto.aux_parameter import Parameter
from NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key import LookUpKey
from NssMPC.crypto.primitives.function_secret_sharing.dicf import SigmaDICFKey


class DivKey(Parameter):
    @ParameterRegistry.ignore()
    class _NegExp2Key(Parameter):
        def __init__(self):
            self.look_up_key = LookUpKey()
            self.table = None

        def __getitem__(self, item):
            key = super(DivKey._NegExp2Key, self).__getitem__(item)
            key.table = self.table
            return key

    def __init__(self):
        self.neg_exp2_key = DivKey._NegExp2Key()
        self.sigma_key = SigmaDICFKey()

    def __len__(self):
        return len(self.sigma_key)

    @staticmethod
    def _gen_neg_exp2_key(num_of_keys):
        down_bound = -SCALE_BIT
        upper_bound = SCALE_BIT + 1

        k0, k1 = DivKey._NegExp2Key(), DivKey._NegExp2Key()

        k0.look_up_key, k1.look_up_key = LookUpKey.gen(num_of_keys, down_bound, upper_bound)
        k0.table = k1.table = _create_neg_exp2_table(down_bound, upper_bound)

        return k0, k1

    @staticmethod
    def gen(num_of_keys):
        k0, k1 = DivKey(), DivKey()
        k0.neg_exp2_key, k1.neg_exp2_key = DivKey._gen_neg_exp2_key(num_of_keys)
        k0.sigma_key, k1.sigma_key = SigmaDICFKey.gen(num_of_keys)

        return k0, k1


def _create_neg_exp2_table(down_bound, upper_bound):
    i = RingTensor.arange(down_bound, upper_bound)
    table = RingTensor.exp2(-i)
    return table
