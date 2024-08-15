"""
computation based on arithmetic secret sharing
client
"""
import unittest

import torch.cuda

from NssMPC.config.configs import DEBUG_LEVEL
from NssMPC.crypto.aux_parameter import B2AKey
from NssMPC.crypto.primitives import ArithmeticSecretSharing
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional.comparison import *
from NssMPC.secure_model.mpc_party import SemiHonestCS
from NssMPC.secure_model.utils.param_provider.param_provider import ParamProvider
from NssMPC.crypto.aux_parameter import MatmulTriples

client = SemiHonestCS(type='client')
client.set_multiplication_provider()
client.append_provider(ParamProvider(param_type=BooleanTriples))
client.append_provider(ParamProvider(param_type=DICFKey))
client.append_provider(ParamProvider(param_type=SigmaDICFKey))
client.append_provider(ParamProvider(param_type=GrottoDICFKey))  # EQUAL
client.append_provider(ParamProvider(param_type=B2AKey))  # B2A
client.set_nonlinear_operation_provider()
client.online()

test_sizes = client.receive()
warmup_times = client.receive()
run_times = client.receive()


class TestClient(unittest.TestCase):

    def setUp(self):
        self.share_x = client.receive()
        self.share_y = client.receive()

    def run_test(self, test_func):
        for i in range(warmup_times + run_times):
            test_func()
            torch.cuda.empty_cache()

    # restoring of x and y
    def test_01_restoring(self):
        print("===================x restore=========================")
        self.run_test(lambda: self.share_x.restore())

    def test_02_addition(self):
        print("===================x + y=========================")
        self.run_test(lambda: self.share_x + self.share_y)

    def test_03_multiplication(self):
        print("===================x * y==========================")
        self.run_test(lambda: self.share_x * self.share_y)

    def test_04_matrix_multiplication(self):
        print("====================x @ y========================")
        # gen beaver triples in advance
        if DEBUG_LEVEL != 2:
            client.providers['MatmulTriples'].param = client.receive()
            client.providers['MatmulTriples'].load_mat_beaver()

        self.run_test(lambda: self.share_x @ self.share_y)

    def test_05_grotto_eq(self):
        print("====================x == y========================")
        self.run_test(lambda: self.share_x == self.share_y)

    def test_06_dicf_ge(self):
        print("====================dicf ge========================")
        self.run_test(lambda: dicf_ge(self.share_x, self.share_y))

    def test_07_grotto_ge(self):
        print("====================ppq ge========================")
        self.run_test(lambda: ppq_ge(self.share_x, self.share_y))

    def test_08_sigma_ge(self):
        print("====================sigma ge========================")
        self.run_test(lambda: sigma_ge(self.share_x, self.share_y))

    def test_09_msb_ge(self):
        print("====================msb ge========================")
        self.run_test(lambda: msb_ge(self.share_x, self.share_y))

    def test_10_div(self):
        print("====================x / y===========================")
        self.run_test(lambda: self.share_x / self.share_y)

    def test_11_exp(self):
        print("======================exp(x)=========================")
        self.run_test(lambda: ArithmeticSecretSharing.exp(self.share_x))

    def test_12_inv_sqrt(self):
        print("======================rsqrt(y)=========================")
        self.run_test(lambda: ArithmeticSecretSharing.rsqrt(self.share_y))


def parametrize(testcase_klass, data_amounts):
    """Generate test cases for different data amounts."""
    test_loader = unittest.TestLoader()
    test_names = test_loader.getTestCaseNames(testcase_klass)
    suite = unittest.TestSuite()
    for name in test_names:
        for _ in data_amounts:
            suite.addTest(testcase_klass(name))
    return suite


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTests(parametrize(TestClient, test_sizes))
    return suite


if __name__ == '__main__':
    unittest.main()
