#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.common.ring.ring_tensor import RingTensor

from NssMPC.config import DEBUG_LEVEL, data_type
from NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key import LookUpKey


class LookUp(object):
    """
    This class defines a look-up table (LUT) used for efficiently calculating values of functions such as `tanh`.
    """

    @staticmethod
    def eval(x, key: LookUpKey, table: RingTensor):
        """
        Uses the look-up table (LUT) to compute function values efficiently.

        This method finds the function value corresponding to `x` in the provided `table`,
        reducing computation overhead compared to direct function calculation.

        :param x: The input parameter for the function.
        :type x: RingTensor
        :param key: The key used in the look-up table algorithm.
        :type key: LookUpKey
        :param table: A `RingTensor` representing the look-up table containing precomputed function values.
        :type table: RingTensor
        :return: The result of the function evaluation, represented as a `RingTensor`.
        :rtype: RingTensor
        """
        shape = x.shape
        x = x.flatten()

        key.phi *= x.scale  # TODO: 临时修正
        key.phi.dtype = x.dtype
        x_shift_shared = key.phi - x
        x_shift = x_shift_shared.restore().convert_to_real_field().to(data_type)
        y = key.onehot_value
        if DEBUG_LEVEL == 2:
            y = y.reshape(1, -1)

        u = x.__class__.rotate(y, shifts=-x_shift)  # TODO: GPU环境下，先转cpu，算完乘法后再转gpu或许更快
        res = (u * table).sum(-1)
        res.dtype = x.dtype

        return res.reshape(shape)
