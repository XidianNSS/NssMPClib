from NssMPC.common.ring.ring_tensor import RingTensor

from NssMPC.config import DEBUG_LEVEL, data_type
from NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key import LookUpKey


class LookUp(object):
    @staticmethod
    def eval(x, key: LookUpKey, table: RingTensor):
        shape = x.shape
        x = x.flatten()

        key.phi.party = x.party
        key.phi *= x.scale  # TODO: 临时修正
        key.phi.dtype = x.dtype
        x_shift_shared = key.phi - x
        x_shift = x_shift_shared.restore().convert_to_real_field().to(data_type)
        y = key.onehot_value
        y.party = x.party
        if DEBUG_LEVEL == 2:
            y = y.reshape(1, -1)

        u = x.__class__.rotate(y, shifts=-x_shift)  # TODO: GPU环境下，先转cpu，算完乘法后再转gpu或许更快
        res = (u * table).sum(-1)
        res.dtype = x.dtype

        return res.reshape(shape)
