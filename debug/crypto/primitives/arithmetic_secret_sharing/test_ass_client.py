"""
computation based on arithmetic secret sharing
client
"""
import unittest

from NssMPC.config.configs import DEBUG_LEVEL
from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.aux_parameter.beaver_triples import MatmulTriples
from NssMPC.crypto.primitives import ArithmeticSecretSharing
from NssMPC.secure_model.mpc_party import SemiHonestCS

client = SemiHonestCS(type='client')
client.set_multiplication_provider()
client.set_comparison_provider()
client.set_nonlinear_operation_provider()
client.online()

x_1 = client.receive()
share_x = x_1

y_1 = client.receive()

share_y = y_1

PartyRuntime(client)

class TestClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        client.close()

    # restoring of x and y
    def test_restoring(self):
        print("===================x restore=========================")
        print('restored x: ', share_x.restore().convert_to_real_field())

        print("===================y restore=========================")
        print('restored y: ', share_y.restore().convert_to_real_field())

    # addition operation
    def test_addition(self):
        print("===================x + y=========================")
        share_z = share_x + share_y
        print('restored x + y: ', share_z.restore().convert_to_real_field())

    # multiplication operation
    def test_multiplication(self):
        print("===================x * y==========================")
        share_z = share_x * share_y
        print('restored x * y: ', share_z.restore().convert_to_real_field())

    # matrix multiplication operation
    def test_matrix_multiplication(self):
        print("====================x @ y========================")
        if DEBUG_LEVEL != 2:
            client.providers[MatmulTriples].param = [client.receive()]
            client.providers[MatmulTriples].load_mat_beaver()

        share_z = share_x @ share_y
        print('restored x @ y: ', share_z.restore().convert_to_real_field())

    def test_equal(self):
        print("====================x == y========================")
        share_z = share_x == share_y
        print('restored x >= y: ', share_z.restore().convert_to_real_field())

    # greater than and equal
    def test_greater_equal(self):
        print("====================x >= y========================")
        share_z = share_x >= share_y
        print('restored x >= y: ', share_z.restore().convert_to_real_field())

    # less than and equal
    def test_less_equal(self):
        print("====================x <= y========================")
        share_z = share_x <= share_y
        print('restored x <= y: ', share_z.restore().convert_to_real_field())

    # greater than
    def test_greater(self):
        print("====================x > y========================")
        share_z = share_x > share_y
        print('restored x > y: ', share_z.restore().convert_to_real_field())

    # less than
    def test_less(self):
        print("====================x < y========================")
        share_z = share_x < share_y
        print('restored x < y: ', share_z.restore().convert_to_real_field())

    def test_div(self):
        print("===================================")
        share_z = share_x / share_y
        print(share_z.restore().convert_to_real_field())
        print("====================================")

    def test_exp(self):
        share_z = ArithmeticSecretSharing.exp(share_x)
        share_z.restore().convert_to_real_field()

        share_z = ArithmeticSecretSharing.exp(share_x)
        share_z = share_z.sum(dim=-1)
        print(share_z.restore().convert_to_real_field())

    def test_inv_sqrt(self):
        share_z = ArithmeticSecretSharing.rsqrt(share_y)
        share_z.restore().convert_to_real_field()
        share_z.restore().convert_to_real_field()
