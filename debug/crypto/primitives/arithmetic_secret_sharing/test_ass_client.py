"""
computation based on arithmetic secret sharing
client
"""
import pytest

from config.base_configs import DEBUG_LEVEL
from crypto.primitives.arithmetic_secret_sharing.arithmetic_shared_ring_tensor import ArithmeticSharedRingTensor
from crypto.primitives.beaver.matrix_triples import MatrixTriples
from model.mpc.semi_honest_party import SemiHonestCS


@pytest.fixture(scope="class", autouse=True)
def my_fixture():
    global client, share_x, share_y
    client = SemiHonestCS(type='client')
    client.set_multiplication_provider()
    client.set_comparison_provider()
    client.connect(('127.0.0.1', 20000), ('127.0.0.1', 20001), ('127.0.0.1', 8089), ('127.0.0.1', 8088))

    x_1 = client.receive()
    share_x = ArithmeticSharedRingTensor(x_1, client)

    y_1 = client.receive()
    share_y = ArithmeticSharedRingTensor(y_1, client)
    yield
    # client.close()


class TestClient:
    # restoring of x and y
    def test_restoring(self, my_fixture):
        print("===================x restore=========================")
        print('restored x: ', share_x.restore().convert_to_real_field())

        print("===================y restore=========================")
        print('restored y: ', share_y.restore().convert_to_real_field())

    # addition operation
    def test_addition(self, my_fixture):
        print("===================x + y=========================")
        share_z = share_x + share_y
        print('restored x + y: ', share_z.restore().convert_to_real_field())

    # multiplication operation
    def test_multiplication(self, my_fixture):
        print("===================x * y==========================")
        share_z = share_x * share_y
        print('restored x * y: ', share_z.restore().convert_to_real_field())

    # matrix multiplication operation
    def test_matrix_multiplication(self, my_fixture):
        print("====================x @ y========================")
        if DEBUG_LEVEL != 2:
            client.providers[MatrixTriples].param = [client.receive()]
            client.providers[MatrixTriples].load_mat_beaver()

        share_z = share_x @ share_y
        print('restored x @ y: ', share_z.restore().convert_to_real_field())

    def test_equal(self, my_fixture):
        print("====================x == y========================")
        share_z = share_x == share_y
        print('restored x >= y: ', share_z.restore().convert_to_real_field())

    # greater than and equal
    def test_greater_equal(self, my_fixture):
        print("====================x >= y========================")
        share_z = share_x >= share_y
        print('restored x >= y: ', share_z.restore().convert_to_real_field())

    # less than and equal
    def test_less_equal(self, my_fixture):
        print("====================x <= y========================")
        share_z = share_x <= share_y
        print('restored x <= y: ', share_z.restore().convert_to_real_field())

    # greater than
    def test_greater(self, my_fixture):
        print("====================x > y========================")
        share_z = share_x > share_y
        print('restored x > y: ', share_z.restore().convert_to_real_field())

    # less than
    def test_less(self, my_fixture):
        print("====================x < y========================")
        share_z = share_x < share_y
        print('restored x < y: ', share_z.restore().convert_to_real_field())
