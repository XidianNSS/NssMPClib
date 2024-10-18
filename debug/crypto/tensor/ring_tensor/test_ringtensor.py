import unittest

import torch

from NssMPC.config.configs import BIT_LEN, DEVICE
from NssMPC.common.ring.ring_tensor import RingTensor


class TestRingTensor(unittest.TestCase):
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], device=DEVICE)
    x = RingTensor([[1, 2, 3], [4, 5, 6]], device=DEVICE)
    y = RingTensor([[1, 2, 3], [4, 5, 6]], device=DEVICE)
    w_max = 2 ** BIT_LEN - 1
    w = 123

    # test the T function in RingTensor
    def test_t(self):
        print("\n-----test the T function in RingTensor---------------\n", self.x.T)
        assert (self.x.T.convert_to_real_field() == RingTensor([[1, 4], [2, 5], [3, 6]]).convert_to_real_field()).all()

    # test the getitem function in RingTensor
    def test_getitem(self):
        print("\n-----test the getitem function in RingTensor---------------\n", self.x.__getitem__(0))
        assert (self.x.__getitem__(0).convert_to_real_field() == RingTensor([1, 2, 3]).convert_to_real_field()).all()

    # test the len function in RingTensor
    def test_len(self):
        print("\n-----test the len function in RingTensor---------------\n", self.x.__len__())
        assert self.x.__len__() == 2

    # test the invert function in RingTensor
    def test_invert(self):
        print("\n-----test the invert function in RingTensor---------------\n", self.x.__invert__())
        assert (self.x.__invert__() == RingTensor([[-2, -3, -4], [-5, -6, -7]])).all()

    # test the add function in RingTensor
    def test_add(self):
        print("\n-----test the add function in RingTensor---------------\n", "other: RingTensor, int, Tensor\n",
              self.x + self.y, self.x + self.w, self.x + self.a)

        res_ring = self.x + self.y
        res_int = self.x + self.w
        res_tensor = self.x + self.a

        ex_res_ring = RingTensor([[2, 4, 6], [8, 10, 12]])
        ex_res_int = RingTensor([[124, 125, 126], [127, 128, 129]])
        ex_res_tensor = RingTensor([[2, 4, 6], [8, 10, 12]])

        assert (res_ring == ex_res_ring).all() and (res_int == ex_res_int).all() \
            and (res_tensor == ex_res_tensor).all()

    # test the radd function in RingTensor
    def test_radd(self):
        print("\n-----test the radd function in RingTensor---------------\n", "other: RingTensor, int, Tensor\n",
              self.x.__radd__(self.y), self.x.__radd__(123), self.x.__radd__(self.a))

        res_ring = self.x.__radd__(self.y)
        res_int = self.x.__radd__(self.w)
        res_tensor = self.x.__radd__(self.a)

        ex_res_ring = RingTensor([[2, 4, 6], [8, 10, 12]])
        ex_res_int = RingTensor([[124, 125, 126], [127, 128, 129]])
        ex_res_tensor = RingTensor([[2, 4, 6], [8, 10, 12]])

        assert (res_ring == ex_res_ring).all() and (res_int == ex_res_int).all() \
               and (res_tensor == ex_res_tensor).all()

    # test the sub function in RingTensor
    def test_sub(self):
        print("\n-----test the sub function in RingTensor---------------\n", "other: RingTensor, int, Tensor\n",
              self.x - self.y, self.x - 123, self.x - self.a)

        res_ring = self.x - self.y
        res_int = self.x - self.w
        res_tensor = self.x - self.a

        ex_res_ring = RingTensor([[0, 0, 0], [0, 0, 0]])
        ex_res_int = RingTensor([[-122, -121, -120], [-119, -118, -117]])
        ex_res_tensor = RingTensor([[0, 0, 0], [0, 0, 0]])

        assert (res_ring == ex_res_ring).all() and (res_int == ex_res_int).all() \
               and (res_tensor == ex_res_tensor).all()

    # test the rsub function in RingTensor
    def test_rsub(self):
        print("\n-----test the rsub function in RingTensor---------------\n", "other: RingTensor, int, Tensor\n",
              self.x.__rsub__(self.y), self.x.__rsub__(self.w), self.x.__rsub__(self.a))

        res_ring = self.x.__rsub__(self.y)
        res_int = self.x.__rsub__(self.w)
        res_tensor = self.x.__rsub__(self.a)

        ex_res_ring = RingTensor([[0, 0, 0], [0, 0, 0]])
        ex_res_int = RingTensor([[122, 121, 120], [119, 118, 117]])
        ex_res_tensor = RingTensor([[0, 0, 0], [0, 0, 0]])

        assert (res_ring == ex_res_ring).all() and (res_int == ex_res_int).all() \
               and (res_tensor == ex_res_tensor).all()

    # test the mul function in RingTensor
    def test_mul(self):
        print("\n-----test the mul function in RingTensor---------------\n", "other: RingTensor, int\n",
              self.x * self.y, self.x * self.w)

        res_ring = self.x * self.y
        res_int = self.x * self.w

        ex_res_ring = RingTensor([[1, 4, 9], [16, 25, 36]])
        ex_res_int = RingTensor([[123, 246, 369], [492, 615, 738]])

        assert (res_ring == ex_res_ring).all() and (res_int == ex_res_int).all()

    # test the rmul function in RingTensor
    def test_rmul(self):
        print("\n-----test the rmul function in RingTensor---------------\n", "other: RingTensor, int\n",
              self.x.__rmul__(self.y), self.x.__rmul__(self.w))

        res_ring = self.x.__rmul__(self.y)
        res_int = self.x.__rmul__(self.w)

        ex_res_ring = RingTensor([[1, 4, 9], [16, 25, 36]])
        ex_res_int = RingTensor([[123, 246, 369], [492, 615, 738]])

        assert (res_ring == ex_res_ring).all() and (res_int == ex_res_int).all()

    # test the mod function in RingTensor
    def test_mod(self):
        print("\n-----test the mod function in RingTensor---------------\n", self.x.__mod__(3))
        assert ((self.x.__mod__(3)) == RingTensor([[1, 2, 0], [1, 2, 0]])).all()

    # test the matmul function in RingTensor
    def test_matmul(self):
        print("\n-----test the matmul function in RingTensor---------------\n", self.x @ self.y.T)
        assert ((self.x @ self.y.T) == RingTensor([[14, 32], [32, 77]])).all()

    # test the truediv function in RingTensor
    def test_truediv(self):
        print("\n-----test the truediv function in RingTensor---------------\n", "other: RingTensor, Tensor, int\n",
              self.x / self.y, self.x / self.a, self.x / 2)

        res_ring = self.x / self.y
        res_tensor = self.x / self.a
        res_int = self.x / 2

        ex_res_ring = RingTensor([[1., 1., 1.], [1., 1., 1.]])
        ex_res_tensor = RingTensor([[1., 1., 1.], [1., 1., 1.]])
        ex_res_int = RingTensor([[0., 1., 2.], [2., 2., 3.]])

        assert (res_ring == ex_res_ring).all() and (res_int == ex_res_int).all() and \
               (res_tensor == ex_res_tensor).all()

    # test the floordiv function in RingTensor
    def test_floordiv(self):
        print("\n-----test the floordiv function in RingTensor---------------\n", "other: RingTensor, Tensor, int\n",
              self.x // self.y, self.x // self.a, self.x // 2)

        res_ring = self.x // self.y
        res_tensor = self.x // self.a
        res_int = self.x // 2

        ex_res_ring = RingTensor([[1, 1, 1], [1, 1, 1]])
        ex_res_tensor = RingTensor([[1, 1, 1], [1, 1, 1]])
        ex_res_int = RingTensor([[0, 1, 1], [2, 2, 3]])

        assert (res_ring == ex_res_ring).all() and (res_int == ex_res_int).all() and \
               (res_tensor == ex_res_tensor).all()

    # test the neg function in RingTensor
    def test_neg(self):
        print("\n-----test the neg function in RingTensor---------------\n", self.x.__neg__())
        assert (self.x.__neg__() == RingTensor([[-1, -2, -3], [-4, -5, -6]])).all()

    # test the eq function in RingTensor
    def test_eq(self):
        print("\n-----test the eq function in RingTensor---------------\n",
              self.x == RingTensor([[1, 2, 3], [4, 5, 6]]))
        assert (self.x == RingTensor([[1, 2, 3], [4, 5, 6]])).all() and (RingTensor(1) == 1).all()

    # test the ne function in RingTensor
    def test_ne(self):
        print("\n-----test the ne function in RingTensor---------------\n",
              self.x != RingTensor([[-1, 2, 3], [4, 5, 6]]))
        assert (self.x != RingTensor([[-1, 2, 3], [4, 5, 6]])).any() and (RingTensor(1) != 2).all()

    # test the gt function in RingTensor
    def test_gt(self):
        print("\n-----test the gt function in RingTensor---------------\n", self.x > RingTensor([[0, 2, 3], [4, 5, 6]]))
        assert ((self.x > RingTensor([[0, 2, 4], [4, 5, 6]])) == RingTensor(
            [[True, False, False], [False, False, False]])).all() and (RingTensor(2) > 1).all()

    # test the ge function in RingTensor
    def test_ge(self):
        print("\n-----test the ge function in RingTensor---------------\n",
              self.x >= RingTensor([[0, 2, 4], [4, 5, 6]]))
        assert ((self.x >= RingTensor([[0, 2, 4], [4, 5, 6]])) == RingTensor(
            [[True, True, False], [True, True, True]])).all() and (RingTensor(2) >= 2).all()

    # test the lt function in RingTensor
    def test_lt(self):
        print("\n-----test the lt function in RingTensor---------------\n", self.x < RingTensor([[0, 3, 3], [4, 5, 6]]))
        assert ((self.x < RingTensor([[0, 3, 3], [4, 5, 6]])) == RingTensor(
            [[False, True, False], [False, False, False]])).all() and (RingTensor(2) < 3).all()

    # test the le function in RingTensor
    def test_le(self):
        print("\n-----test the le function in RingTensor---------------\n",
              self.x <= RingTensor([[0, 2, 4], [4, 5, 6]]))
        assert ((self.x <= RingTensor([[0, 2, 4], [4, 5, 6]])) == RingTensor(
            [[False, True, True], [True, True, True]])).all() and (RingTensor(3) <= 3).all()

    # test the xor function in RingTensor
    def test_xor(self):
        print("\n-----test the xor function in RingTensor---------------\n", self.x ^ self.y.__neg__())
        assert ((self.x ^ self.y.__neg__()) == RingTensor([[-2, -4, -2], [-8, -2, -4]])).all()

    # test the or function in RingTensor
    def test_or(self):
        print("\n-----test the or function in RingTensor---------------\n", self.x | self.y.__neg__(), self.x | 3)
        assert ((self.x | self.y.__neg__()) == RingTensor([[-1, -2, -1], [-4, -1, -2]])).all() and (
                self.x | 3 == RingTensor([[3, 3, 3], [7, 7, 7]])).all()

    # test the and function in RingTensor
    def test_and(self):
        print("\n-----test the and function in RingTensor---------------\n", self.x & self.y.__neg__(), self.x & 3)
        assert ((self.x & self.y.__neg__()) == RingTensor([[1, 2, 1], [4, 1, 2]])).all() and (
                self.x & 3 == RingTensor([[1, 2, 3], [0, 1, 2]])).all()

    # test the rshift function in RingTensor
    def test_rshift(self):
        print("\n-----test the rshift function in RingTensor---------------\n", self.x >> 1)
        assert ((self.x >> 1) == RingTensor([[0, 1, 1], [2, 2, 3]])).all()

    # test the lshift function in RingTensor
    def test_lshift(self):
        print("\n-----test the lshift function in RingTensor---------------\n", self.x << 2)
        assert ((self.x << 2) == RingTensor([[4, 8, 12], [16, 20, 24]])).all()

    # test the cat function in RingTensor
    def test_cat(self):
        print("\n-----test the cat function in RingTensor---------------\n", self.x.cat(self.y, 0))
        assert (self.x.cat(self.y, 0) == RingTensor([1, 2, 3, 4, 5, 6])).all()

    # test the stack function in RingTensor
    def test_stack(self):
        print("\n-----test the stack function in RingTensor---------------\n", self.x.stack([self.x, self.y], 0))
        assert (self.x.stack([self.x, self.y], 0) == RingTensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])).all()

    # test the clone function in RingTensor
    def test_clone(self):
        print("\n-----test the clone function in RingTensor---------------\n", self.x.clone())
        assert (self.x.clone() == RingTensor([[1, 2, 3], [4, 5, 6]])).all()

    # test the get_bit function in RingTensor
    def test_get_bit(self):
        print("\n-----test the get_bit function in RingTensor---------------\n", self.x.get_bit(1))
        assert (self.x.get_bit(1) == RingTensor([[0, 1, 1], [0, 0, 1]])).all()

    # test the reshape function in RingTensor
    def test_reshape(self):
        print("\n-----test the reshape function in RingTensor---------------\n", self.x.reshape((3, 2)))
        assert (self.x.reshape((3, 2)) == RingTensor([[1, 2], [3, 4], [5, 6]])).all()

    # test the img2col function in RingTensor
    def test_img2col(self):
        x = RingTensor([[[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]], [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]]])
        print("\n-----test the img2col function in RingTensor---------------\n", x.img2col(2, 1))
        assert x.img2col(2, 1)[1:] == (1, 6, 2)

    # test the size function in RingTensor
    def test_size(self):
        x = RingTensor([[[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]], [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]]])
        print("\n-----test the size function in RingTensor---------------\n", x.size())
        assert x.size() == torch.Size([1, 2, 3, 4])

    # test the view function in RingTensor
    def test_view(self):
        print("\n-----test the view function in RingTensor---------------\n", self.x.view((3, 2)))
        assert (self.x.view((3, 2)) == RingTensor([[1, 2], [3, 4], [5, 6]])).all()

    # test the flatten function in RingTensor
    def test_flatten(self):
        print("\n-----test the flatten function in RingTensor---------------\n", self.x.flatten())
        assert ((self.x.flatten()) == RingTensor([1, 2, 3, 4, 5, 6])).all()

    # test the permute function in RingTensor
    def test_permute(self):
        print("\n-----test the permute function in RingTensor---------------\n", self.x.permute((1, 0)))
        assert ((self.x.permute((1, 0))) == RingTensor([[1, 4], [2, 5], [3, 6]])).all()

    # test the tolist function in RingTensor
    def test_tolist(self):
        print("\n-----test the tolist function in RingTensor---------------\n", self.x.tolist())
        assert (self.x.tolist()) == [[1, 2, 3], [4, 5, 6]]

    # test the numel function in RingTensor
    def test_numel(self):
        print("\n-----test the numel function in RingTensor---------------\n", self.x.numel())
        assert (self.x.numel()) == 6

    # test the signbit function in RingTensor
    def test_signbit(self):
        x = RingTensor([[1, 2, -3], [-1, 2, -3]])
        print("\n-----test the signbit function in RingTensor---------------\n", x.signbit())
        assert ((x.signbit()) == RingTensor([[0, 0, 1], [1, 0, 1]])).all()

    # test the bit_slice function in RingTensor
    def test_bit_slice(self):
        print("\n-----test the bit_slice function in RingTensor---------------\n", self.x.bit_slice(1, 2))
        assert (self.x.bit_slice(1, 2) == RingTensor([[0, 1, 1], [2, 2, 3]])).all()
