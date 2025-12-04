#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
The implementation is based on the work of `Damgård I, Nielsen J B, Nielsen M, et al. The tinytable protocol for 2-party secure computation, or: Gate-scrambling revisited 2017: 167-187.`.
For reference, see the `paper <https://eprint.iacr.org/2016/695.pdf>`_.
"""
from typing import Tuple

from NssMPC.infra.mpc.aux_parameter.parameter import Parameter, ParameterRegistry
from NssMPC.infra.tensor import RingTensor
from NssMPC.primitives.secret_sharing import AdditiveSecretSharing


class SelectLin(object):
    @staticmethod
    def eval(x_shift: RingTensor, w_shift: RingTensor, d_shift: RingTensor, key: 'SelectLinKey') -> RingTensor:
        shape = x_shift.shape
        i_shift = (w_shift * 2 + d_shift) % 4

        key.p.dtype = x_shift.dtype
        key.q.dtype = x_shift.dtype

        p = key.p.__class__.gather(key.p, -1, i_shift.tensor.unsqueeze(1)).squeeze() if len(key.p.shape) == 2 else \
            key.p[i_shift.tensor]
        q = key.p.__class__.gather(key.q, -1, i_shift.tensor.unsqueeze(1)).squeeze() if len(key.p.shape) == 2 else \
            key.q[i_shift.tensor]
        return (p * x_shift.flatten() + q).reshape(shape)

    @staticmethod
    def eval_with_comm(x_shift: RingTensor, w: RingTensor, d: RingTensor, key: 'SelectLinKey') -> RingTensor:
        w_shift = AdditiveSecretSharing(key.w) + w.flatten()
        d_shift = AdditiveSecretSharing(key.d) + d.flatten()
        length = w_shift.numel()
        w_and_d = AdditiveSecretSharing.cat([w_shift, d_shift], dim=0).restore()
        w_shift = w_and_d[:length]
        d_shift = w_and_d[length:]
        return SelectLin.eval(x_shift, w_shift, d_shift, key)


@ParameterRegistry.ignore()
class SelectLinKey(Parameter):
    def __init__(self):
        self.p = None
        self.q = None
        self.w = None
        self.d = None

    @staticmethod
    def gen(num_of_keys: int, p: RingTensor, q: RingTensor) -> Tuple['SelectLinKey', 'SelectLinKey']:
        w = RingTensor.random([num_of_keys], down_bound=0, upper_bound=2)
        d = RingTensor.random([num_of_keys], down_bound=0, upper_bound=2)
        i = (w * 2 + d) % 4

        p = RingTensor.rotate(p, i.tolist())
        q = RingTensor.rotate(q, i.tolist())

        k0 = SelectLinKey()
        k1 = SelectLinKey()

        k0.p, k1.p = AdditiveSecretSharing.share(p, 2)
        k0.q, k1.q = AdditiveSecretSharing.share(q, 2)

        k0.w, k1.w = AdditiveSecretSharing.share(w, 2)
        k0.d, k1.d = AdditiveSecretSharing.share(d, 2)

        k0.w = k0.w.item
        k1.w = k1.w.item
        k0.d = k0.d.item
        k1.d = k1.d.item

        return k0, k1
