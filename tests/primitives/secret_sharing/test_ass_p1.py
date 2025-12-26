"""
computation based on arithmetic secret sharing
party 1
"""
import unittest

import nssmpc
from nssmpc import Party2PC, PartyRuntime, SEMI_HONEST
from nssmpc.config.configs import DEBUG_LEVEL
from nssmpc.protocols.semi_honest_2pc.multiplication import MatmulTriples

party = Party2PC(1, SEMI_HONEST)
party.online()
with PartyRuntime(party):
    share_x = nssmpc.SecretTensor(src_id=0)
    share_y = nssmpc.SecretTensor(src_id=0)


class TestParty1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = PartyRuntime(party)
        cls.ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        party.close()
        cls.ctx.__exit__(None, None, None)

    # restoring of x and y
    def test_restoring(self):
        print("===================x restore=========================")
        print('restored x: ', share_x.recon().convert_to_real_field())

        print("===================y restore=========================")
        print('restored y: ', share_y.recon().convert_to_real_field())

    # addition operation
    def test_addition(self):
        print("===================x + y=========================")
        share_z = share_x + share_y
        print('restored x + y: ', share_z.recon().convert_to_real_field())

    # multiplication operation
    def test_multiplication(self):
        print("===================x * y==========================")
        share_z = share_x * share_y
        print('restored x * y: ', share_z.recon().convert_to_real_field())

    # matrix multiplication operation
    def test_matrix_multiplication(self):
        print("====================x @ y========================")
        share_z = share_x @ share_y
        print('restored x @ y: ', share_z.recon().convert_to_real_field())

    def test_equal(self):
        print("====================x == y========================")
        share_z = share_x == share_y
        print('restored x >= y: ', share_z.recon().convert_to_real_field())

    # greater than and equal
    def test_greater_equal(self):
        print("====================x >= y========================")
        share_z = share_x >= share_y
        print('restored x >= y: ', share_z.recon().convert_to_real_field())

    # less than and equal
    def test_less_equal(self):
        print("====================x <= y========================")
        share_z = share_x <= share_y
        print('restored x <= y: ', share_z.recon().convert_to_real_field())

    # greater than
    def test_greater(self):
        print("====================x > y========================")
        share_z = share_x > share_y
        print('restored x > y: ', share_z.recon().convert_to_real_field())

    # less than
    def test_less(self):
        print("====================x < y========================")
        share_z = share_x < share_y
        print('restored x < y: ', share_z.recon().convert_to_real_field())

    def test_div(self):
        print("===================================")
        share_z = share_x / share_y
        print(share_z.recon().convert_to_real_field())
        print("====================================")

    def test_exp(self):
        share_z = share_x.exp()
        share_z.recon().convert_to_real_field()

        share_z = share_x.exp()
        share_z = share_z.sum(dim=-1)
        print(share_z.recon().convert_to_real_field())

    def test_inv_sqrt(self):
        share_z = share_y.rsqrt()
        share_z.recon().convert_to_real_field()
        share_z.recon().convert_to_real_field()
