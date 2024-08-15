import torch
from NssMPC import RingTensor
from NssMPC.config import SCALE_BIT, data_type
from NssMPC.crypto.aux_parameter import Parameter
from NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key import LookUpKey
from NssMPC.crypto.primitives.function_secret_sharing.dicf import SigmaDICFKey


class ReciprocalSqrtKey(Parameter):
    def __init__(self):
        self.neg_exp2_look_up_key = LookUpKey()
        self.rec_sqrt_look_up_key = LookUpKey()
        self.sigma_key = SigmaDICFKey()
        self.neg_exp2_table = None
        self.rec_sqrt_table = None

    def __getitem__(self, item):
        key = super(ReciprocalSqrtKey, self).__getitem__(item)
        key.neg_exp2_table = self.neg_exp2_table
        key.rec_sqrt_table = self.rec_sqrt_table
        return key

    def __len__(self):
        return len(self.sigma_key)

    @staticmethod
    def gen(num_of_keys, in_scale_bit=SCALE_BIT, out_scale_bit=SCALE_BIT):
        down_bound = -SCALE_BIT
        upper_bound = SCALE_BIT + 1

        neg_exp2_table = _create_neg_exp2_table(down_bound, upper_bound)
        rec_sqrt_table = _create_rsqrt_table(in_scale_bit, out_scale_bit)
        k0, k1 = ReciprocalSqrtKey(), ReciprocalSqrtKey()
        k0.neg_exp2_table = k1.neg_exp2_table = neg_exp2_table
        k0.rec_sqrt_table = k1.rec_sqrt_table = rec_sqrt_table

        k0.neg_exp2_look_up_key, k1.neg_exp2_look_up_key = LookUpKey.gen(num_of_keys, down_bound, upper_bound)
        k0.rec_sqrt_look_up_key, k1.rec_sqrt_look_up_key = LookUpKey.gen(num_of_keys, 0, 8191)
        k0.sigma_key, k1.sigma_key = SigmaDICFKey.gen(num_of_keys)

        return k0, k1


def _create_rsqrt_table(in_scale_bit=SCALE_BIT, out_scale_bit=SCALE_BIT):
    i = torch.arange(0, 2 ** 13 - 1, dtype=torch.float64)
    e = i % 64
    m = i // 64
    q = 2 ** e * (1 + m / 128)

    rec_sqrt_table = torch.sqrt(2 ** in_scale_bit / q) * 2 ** out_scale_bit
    rec_sqrt_table = rec_sqrt_table.to(data_type)
    rec_sqrt_table = RingTensor(rec_sqrt_table, 'float')

    return rec_sqrt_table


def _create_neg_exp2_table(down_bound, upper_bound):
    i = RingTensor.arange(down_bound, upper_bound)
    table = RingTensor.exp2(-i)
    return table
