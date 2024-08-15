import torch
from NssMPC import RingTensor
from NssMPC.config import data_type, DEVICE, SCALE_BIT, BIT_LEN, HALF_RING, GELU_TABLE_BIT
from NssMPC.config.runtime import ParameterRegistry
from NssMPC.crypto.aux_parameter import SigmaDICFKey, DPFKey, Parameter
from NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key import LookUpKey
from NssMPC.crypto.aux_parameter.select_keys import SelectLinKey


class GeLUKey(Parameter):
    @ParameterRegistry.ignore()
    class _SelectKey(Parameter):
        def __init__(self):
            self.w = None
            self.z = None

    def __init__(self):
        self.look_up_key = LookUpKey()
        self.sigma_key = SigmaDICFKey()
        self.select_lin_key = SelectLinKey()
        self.select_key = GeLUKey._SelectKey()
        self.look_up_table = None

    def __len__(self):
        return len(self.sigma_key)

    @staticmethod
    def _gen_select_key(s_r_in, x_r_in):
        w = s_r_in * x_r_in
        z = 2 * s_r_in * x_r_in

        k0, k1 = GeLUKey._SelectKey(), GeLUKey._SelectKey()
        from NssMPC import ArithmeticSecretSharing
        k0.w, k1.w = ArithmeticSecretSharing.share(w, 2)
        k0.z, k1.z = ArithmeticSecretSharing.share(z, 2)

        k0.w, k1.w, k0.z, k1.z = k0.w.item, k1.w.item, k0.z.item, k1.z.item
        return k0, k1

    @staticmethod
    def _gen_sigma_cmp_key(num_of_keys, table_scale_bit):
        k0, k1 = SigmaDICFKey(), SigmaDICFKey()

        input_r_in = RingTensor.random([num_of_keys], down_bound=-HALF_RING // 2, upper_bound=HALF_RING // 2)
        from NssMPC import ArithmeticSecretSharing
        k0.r_in, k1.r_in = ArithmeticSecretSharing.share(input_r_in, 2)
        k0.r_in, k1.r_in = k0.r_in.item, k1.r_in.item
        small_r = input_r_in // 2 ** (SCALE_BIT - table_scale_bit)
        small_r.bit_len = BIT_LEN - SCALE_BIT + table_scale_bit

        y1 = small_r % (HALF_RING - 1)
        k0.dpf_key, k1.dpf_key = DPFKey.gen(num_of_keys, y1, RingTensor(1))
        c = input_r_in.signbit()
        c0 = RingTensor.random([num_of_keys], down_bound=0, upper_bound=2)
        c1 = c ^ c0

        k0.c = c0
        k1.c = c1

        return k0, k1, input_r_in, small_r

    @staticmethod
    def gen(num_of_keys):
        """

        the default scale of elements in the gelu table is 2 ** 6

        Args:
            num_of_keys: the number of keys
        Returns:

        """
        table_scale_bit = GELU_TABLE_BIT
        table_scale = 2 ** table_scale_bit
        table_size = 4 * table_scale
        k0 = GeLUKey()
        k1 = GeLUKey()

        k0.look_up_key, k1.look_up_key = LookUpKey.gen(num_of_keys, 0, table_size)
        k0.look_up_table = k1.look_up_table = _create_gelu_table(table_scale_bit)

        k0.sigma_key, k1.sigma_key, input_r_in, select_lin_r = GeLUKey._gen_sigma_cmp_key(num_of_keys, table_scale_bit)

        p = RingTensor([0, 0, -1, 1]).repeat(num_of_keys, 1)

        q = RingTensor([2 ** (table_scale_bit + 2) - 1, 2 ** (table_scale_bit + 2) - 1]).repeat(num_of_keys, 1)
        q = RingTensor.cat((q, select_lin_r.view(-1, 1), (-select_lin_r).view(-1, 1)), dim=1)

        k0.select_lin_key, k1.select_lin_key = SelectLinKey.gen(num_of_keys, p, q)

        r_in_1 = (k0.select_lin_key.d + k1.select_lin_key.d) % 2

        k0.select_key, k1.select_key = GeLUKey._gen_select_key(r_in_1, input_r_in)

        return k0, k1

    def __getitem__(self, item):
        key = super(GeLUKey, self).__getitem__(item)
        key.look_up_table = self.look_up_table
        return key


def _create_gelu_table(table_scale_bit=GELU_TABLE_BIT):
    """
    In the input domain range, ReLU(x)-GeLU(x) is non-zero on (-4, 4), and is an even function,
    so the input range can be determined to be [0, 4)

    The precision of this input is determined by table_scale_bit(f), that is, the input range [0, 2 ** (f + 2))
    Args:
        table_scale_bit:

    Returns: the table of the ReLU(x)-GeLU(x)
    """

    table_scale = 2 ** table_scale_bit
    table_key = torch.arange(0, 4 * table_scale, dtype=data_type, device=DEVICE) / table_scale
    relu = torch.nn.ReLU()
    gelu = torch.nn.GELU()

    table_norm_value = relu(table_key) - gelu(table_key)
    table = RingTensor.convert_to_ring(table_norm_value)
    return table
