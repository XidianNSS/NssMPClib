"""
computation based on arithmetic secret sharing
server
"""
import unittest

from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.aux_parameter.beaver_triples import MatmulTriples

from NssMPC import ArithmeticSecretSharing
from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.config.configs import *
from NssMPC.secure_model.mpc_party import SemiHonestCS

server = SemiHonestCS(type='server')
server.set_multiplication_provider()
server.set_comparison_provider()
server.set_nonlinear_operation_provider()
server.online()
#
x = torch.rand([10, 10]).to(DEVICE)
y = torch.rand([10, 10]).to(DEVICE)
# x = torch.randint(0, 10, [10, 10]).to(DEVICE)
# y = torch.randint(0, 10, [10, 10]).to(DEVICE)
x_ring = RingTensor.convert_to_ring(x)

x_0, x_1 = ArithmeticSecretSharing.share(x_ring, 2)
server.send(x_1)
share_x = x_0

y_ring = RingTensor.convert_to_ring(y)

y_0, y_1 = ArithmeticSecretSharing.share(y_ring, 2)
server.send(y_1)
share_y = y_0

PartyRuntime(server)


class TestServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        server.close()

    # restoring of x and y
    def test_restoring(self):
        print("===================x restore=========================")
        print('origin x: ', x)
        res_x = share_x.restore().convert_to_real_field()
        print('restored x: ', res_x)
        assert torch.allclose(x.to(res_x), res_x, atol=1e-3, rtol=1e-3) == True

        print("===================y restore=========================")
        print('origin y: ', y)
        res_y = share_y.restore().convert_to_real_field()
        print('restored y: ', res_y)

        assert torch.allclose(y.to(res_y), res_y, atol=1e-3, rtol=1e-3) == True

    # addition operation
    def test_addition(self):
        print("===================x + y=========================")
        print('x + y: ', x + y)
        share_z = share_x + share_y
        res_share_z = share_z.restore().convert_to_real_field()
        print('restored x + y: ', res_share_z)
        assert torch.allclose((x + y).to(res_share_z), res_share_z, atol=1e-3, rtol=1e-3) == True

    # multiplication operation
    def test_multiplication(self):
        print("===================x * y==========================")
        print('x * y: ', x * y)
        print(x)
        share_z = share_x * share_y
        res_share_z = share_z.restore().convert_to_real_field()
        print('restored x * y: ', res_share_z)
        assert torch.allclose((x * y).to(res_share_z), res_share_z, atol=1e-3, rtol=1e-3) == True

    # matrix multiplication operation
    def test_matrix_multiplication(self):
        print("====================x @ y========================")
        # gen beaver triples in advance
        if DEBUG_LEVEL != 2:
            triples = MatmulTriples.gen(1, x.shape, y.shape)
            server.providers[MatmulTriples].param = [triples[0]]
            server.send(triples[1])
            server.providers[MatmulTriples].load_mat_beaver()

        print('x @ y: ', x @ y)
        share_z = share_x @ share_y
        res_share_z = share_z.restore().convert_to_real_field()
        print('restored x @ y: ', res_share_z)
        assert torch.allclose((x @ y).to(res_share_z), res_share_z, atol=1e-3, rtol=1e-3) == True

    # equal
    def test_equal(self):
        print("====================x == y========================")
        print('x == y: ', x == y)
        share_z = share_x == share_y
        res_share_z = share_z.restore().convert_to_real_field()
        print('restored x == y: ', res_share_z)
        assert torch.allclose((x == y).to(res_share_z), res_share_z, atol=1e-3, rtol=1e-3) == True

    # greater than and equal
    def test_greater_equal(self):
        print("====================x >= y========================")
        print('x >= y: ', x >= y)
        share_z = share_x >= share_y
        res_share_z = share_z.restore().convert_to_real_field()
        print('restored x >= y: ', res_share_z)
        assert torch.allclose((x >= y).to(res_share_z), res_share_z, atol=1e-3, rtol=1e-3) == True

    # less than and equal
    def test_less_equal(self):
        print("====================x <= y========================")
        print('x <= y: ', x <= y)
        share_z = share_x <= share_y
        res_share_z = share_z.restore().convert_to_real_field()
        print('restored x <= y: ', res_share_z)
        assert torch.allclose((x <= y).to(res_share_z), res_share_z, atol=1e-3, rtol=1e-3) == True

    # greater than
    def test_greater(self):
        print("====================x > y========================")
        print('x > y: ', x > y)
        share_z = share_x > share_y
        res_share_z = share_z.restore().convert_to_real_field()
        print('restored x > y: ', res_share_z)
        assert torch.allclose((x > y).to(res_share_z), res_share_z, atol=1e-3, rtol=1e-3) == True

    # less than
    def test_less(self):
        print("====================x < y========================")
        print('x < y: ', x < y)
        share_z = share_x < share_y
        res_share_z = share_z.restore().convert_to_real_field()
        print('restored x < y: ', res_share_z)
        assert torch.allclose((x < y).to(res_share_z), res_share_z, atol=1e-3, rtol=1e-3) == True

    def test_div(self):
        print("===============================================")
        print(x / y)
        share_z = share_x / share_y
        res_share_z = share_z.restore().convert_to_real_field()
        assert torch.allclose((x / y).to(res_share_z), res_share_z, atol=1e-3, rtol=1e-3) == True
        print("================================================")

    def test_exp(self):
        print("===============================================")
        print("明文数据", torch.exp(x))
        share_z = ArithmeticSecretSharing.exp(share_x)
        res_share_z = share_z.restore().convert_to_real_field()
        print(res_share_z)
        assert torch.allclose(torch.exp(x).to(res_share_z), res_share_z, atol=5e-1, rtol=5e-1) == True
        print("===============================================")

        print("明文数据先exp再求和", torch.sum(torch.exp(x), dim=-1))
        share_z = ArithmeticSecretSharing.exp(share_x)
        share_z = share_z.sum(dim=-1)
        res_share_z = share_z.restore().convert_to_real_field()

        assert torch.allclose(torch.sum(torch.exp(x), dim=-1).to(res_share_z), res_share_z, atol=5e-1,
                              rtol=5e-1) == True
        print("===============================================")

    def test_inv_sqrt(self):
        print("===============================================")
        print("明文数据", torch.rsqrt(y))
        share_z = ArithmeticSecretSharing.rsqrt(share_y)
        print(share_z.restore().convert_to_real_field())
        res_share_z = torch.max((torch.rsqrt(y)) - share_z.restore().convert_to_real_field())
        assert torch.allclose(torch.rsqrt(y).to(res_share_z), res_share_z, atol=5e-3, rtol=5e-3) == True
        print("===============================================")
