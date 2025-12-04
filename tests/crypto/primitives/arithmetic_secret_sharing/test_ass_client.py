"""
computation based on arithmetic secret sharing
client
"""
import unittest

import NssMPC
from NssMPC.config.configs import DEBUG_LEVEL
from NssMPC.infra.mpc.party import PartyCtx, Party2PC

from NssMPC.protocols.semi_honest_2pc.multiplication import MatmulTriples
from NssMPC.primitives.secret_sharing import AdditiveSecretSharing
from NssMPC.runtime.presets import SEMI_HONEST
from NssMPC.runtime.context import PartyRuntime

client = Party2PC(1, SEMI_HONEST)
client.online()
with PartyRuntime(client):
    share_x=NssMPC.SecretTensor(src_id=0)
    share_y=NssMPC.SecretTensor(src_id=0)

    PartyCtx.set(client)


class TestClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = PartyRuntime(client)
        cls.ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        client.close()
        cls.ctx.__exit__(None, None, None)

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
            client.providers[MatmulTriples].param = [client.recv()]
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
        share_z = AdditiveSecretSharing.exp(share_x)
        share_z.restore().convert_to_real_field()

        share_z = AdditiveSecretSharing.exp(share_x)
        share_z = share_z.sum(dim=-1)
        print(share_z.restore().convert_to_real_field())

    def test_inv_sqrt(self):
        share_z = AdditiveSecretSharing.rsqrt(share_y)
        share_z.restore().convert_to_real_field()
        share_z.restore().convert_to_real_field()
