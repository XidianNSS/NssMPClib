"""
function_secret_sharing 测试
"""
from common.tensor.ring_tensor import RingTensor
from crypto.protocols.function_secret_sharing import *

num_of_keys = 10  # We need a few keys for a few function values, but of course we can generate many keys in advance.

# generate keys in offline phase
# set alpha and beta
alpha = RingTensor(5)
beta = RingTensor(1)
down_bound = RingTensor(3)
upper_bound = RingTensor(7)
# online phase
# generate some values what we need to evaluate
x = RingTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


class Test:
    # distributed interval containment function
    def test_dicf(self):
        key0, key1 = DICF.gen(num_of_keys=num_of_keys, down_bound=down_bound, upper_bound=upper_bound)

        x_shift = x + key0.r_in.reshape(x.shape) + key1.r_in.reshape(x.shape)

        # online phase
        # Party 0:
        res_0 = DICF.eval(x_shift=x_shift, keys=key0, party_id=0, down_bound=down_bound, upper_bound=upper_bound)
        # Party 1:
        res_1 = DICF.eval(x_shift=x_shift, keys=key1, party_id=1, down_bound=down_bound, upper_bound=upper_bound)

        # restore result
        res = res_0 + res_1
        print(res)
        assert (res == RingTensor([0, 0, 1, 1, 1, 1, 1, 0, 0, 0], device=res.device)).all()

    def test_grotto(self):
        key0, key1 = PPQCompare.gen(num_of_keys=num_of_keys, beta=beta)

        x_shift = key0.r_in.reshape(x.shape) + key1.r_in.reshape(x.shape) - x

        # online phase
        # Party 0:
        res_0 = PPQCompare.eval(x_shift=x_shift, key=key0, party_id=0, down_bound=down_bound,
                                upper_bound=upper_bound)
        # Party 1:
        res_1 = PPQCompare.eval(x_shift=x_shift, key=key1, party_id=1, down_bound=down_bound,
                                upper_bound=upper_bound)

        # restore result
        res = res_0 ^ res_1
        print(res)
        assert (res == RingTensor([0, 0, 1, 1, 1, 1, 0, 0, 0, 0], device=res.device)).all()

    def test_sigma(self):
        from crypto.protocols.function_secret_sharing.sigma import SigmaCompare
        key0, key1 = SigmaCompare.gen(num_of_keys=num_of_keys)

        x_shift = key0.r_in.reshape(x.shape) + key1.r_in.reshape(x.shape) + x - 5

        # online phase
        # Party 0:
        res_0 = SigmaCompare.eval(x_shift=x_shift, key=key0, party_id=0)
        # Party 1:
        res_1 = SigmaCompare.eval(x_shift=x_shift, key=key1, party_id=1)

        # restore result
        res = res_0 ^ res_1
        print(res)
        assert (res == RingTensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], device=res.device)).all()
