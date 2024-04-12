"""
computation based on arithmetic secret sharing
server
"""
import pytest

from common.tensor.ring_tensor import RingTensor
from config.base_configs import *
from crypto.primitives.arithmetic_secret_sharing.arithmetic_shared_ring_tensor import ArithmeticSharedRingTensor
from crypto.primitives.beaver.matrix_triples import MatrixTriples
from model.mpc.semi_honest_party import SemiHonestCS


@pytest.fixture(scope="class", autouse=True)
def my_fixture():
    global server, x, y, share_x, share_y
    server = SemiHonestCS(type='server')
    server.set_multiplication_provider()
    server.set_comparison_provider()
    server.connect(('127.0.0.1', 8089), ('127.0.0.1', 8088), ('127.0.0.1', 20000), ('127.0.0.1', 20001))
    #
    x = torch.rand([10, 10]).to(DEVICE)
    y = torch.rand([10, 10]).to(DEVICE)
    # x = torch.randint(0, 10, [10, 10]).to(DEVICE)
    # y = torch.randint(0, 10, [10, 10]).to(DEVICE)
    x_ring = RingTensor.convert_to_ring(x)

    x_0, x_1 = ArithmeticSharedRingTensor.share(x_ring, 2)
    server.send(x_1)
    share_x = ArithmeticSharedRingTensor(x_0, server)

    y_ring = RingTensor.convert_to_ring(y)

    y_0, y_1 = ArithmeticSharedRingTensor.share(y_ring, 2)
    server.send(y_1)
    share_y = ArithmeticSharedRingTensor(y_0, server)
    yield
    # server.close()


class TestServer:
    # restoring of x and y
    def test_restoring(self, my_fixture):
        print("===================x restore=========================")
        print('origin x: ', x)
        res_x = share_x.restore().convert_to_real_field()
        print('restored x: ', res_x)
        assert torch.allclose(x + .0, res_x, atol=1e-3, rtol=1e-3) == True

        print("===================y restore=========================")
        print('origin y: ', y)
        res_y = share_y.restore().convert_to_real_field()
        print('restored y: ', res_y)
        assert torch.allclose(y + .0, res_y, atol=1e-3, rtol=1e-3) == True

    # addition operation
    def test_addition(self, my_fixture):
        print("===================x + y=========================")
        print('x + y: ', x + y)
        share_z = share_x + share_y
        res_share_z = share_z.restore().convert_to_real_field()
        print('restored x + y: ', res_share_z)
        assert torch.allclose(x + y + .0, res_share_z, atol=1e-3, rtol=1e-3) == True

    # multiplication operation
    def test_multiplication(self, my_fixture):
        print("===================x * y==========================")
        print('x * y: ', x * y)
        share_z = share_x * share_y
        res_share_z = share_z.restore().convert_to_real_field()
        print('restored x * y: ', res_share_z)
        assert torch.allclose(x * y + .0, res_share_z, atol=1e-3, rtol=1e-3) == True

    # matrix multiplication operation
    def test_matrix_multiplication(self, my_fixture):
        print("====================x @ y========================")
        # gen beaver triples in advance
        if DEBUG_LEVEL != 2:
            triples = MatrixTriples.gen(1, x.shape, y.shape)
            server.providers[MatrixTriples].param = [triples[0]]
            server.send(triples[1])
            server.providers[MatrixTriples].load_mat_beaver()

        print('x @ y: ', x @ y)
        share_z = share_x @ share_y
        res_share_z = share_z.restore().convert_to_real_field()
        print('restored x @ y: ', res_share_z)
        assert torch.allclose(x @ y + .0, res_share_z, atol=1e-3, rtol=1e-3) == True

    # equal
    def test_equal(self, my_fixture):
        print("====================x == y========================")
        print('x == y: ', x == y)
        share_z = share_x == share_y
        res_share_z = share_z.restore().convert_to_real_field()
        print('restored x >= y: ', res_share_z)
        assert torch.allclose((x == y) + .0, res_share_z, atol=1e-3, rtol=1e-3) == True

    # greater than and equal
    def test_greater_equal(self, my_fixture):
        print("====================x >= y========================")
        print('x >= y: ', x >= y)
        share_z = share_x >= share_y
        res_share_z = share_z.restore().convert_to_real_field()
        print('restored x >= y: ', res_share_z)
        assert torch.allclose((x >= y) + .0, res_share_z, atol=1e-3, rtol=1e-3) == True

    # less than and equal
    def test_less_equal(self, my_fixture):
        print("====================x <= y========================")
        print('x <= y: ', x <= y)
        share_z = share_x <= share_y
        res_share_z = share_z.restore().convert_to_real_field()
        print('restored x <= y: ', res_share_z)
        assert torch.allclose((x <= y) + .0, res_share_z, atol=1e-3, rtol=1e-3) == True

    # greater than
    def test_greater(self, my_fixture):
        print("====================x > y========================")
        print('x > y: ', x > y)
        share_z = share_x > share_y
        res_share_z = share_z.restore().convert_to_real_field()
        print('restored x > y: ', res_share_z)
        assert torch.allclose((x > y) + .0, res_share_z, atol=1e-3, rtol=1e-3) == True

    # less than
    def test_less(self, my_fixture):
        print("====================x < y========================")
        print('x < y: ', x < y)
        share_z = share_x < share_y
        res_share_z = share_z.restore().convert_to_real_field()
        print('restored x < y: ', res_share_z)
        assert torch.allclose((x < y) + .0, res_share_z, atol=1e-3, rtol=1e-3) == True
