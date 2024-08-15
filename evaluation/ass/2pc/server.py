import csv
import time
import unittest

import torch

from NssMPC import ArithmeticSecretSharing
from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.config.configs import *
from NssMPC.crypto.aux_parameter import *
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional.comparison import dicf_ge, ppq_ge, \
    sigma_ge, msb_ge
from NssMPC.secure_model.mpc_party import SemiHonestCS
from NssMPC.secure_model.utils.param_provider.param_provider import ParamProvider

server = SemiHonestCS(type='server')
server.set_multiplication_provider()
server.append_provider(ParamProvider(param_type=BooleanTriples))
server.append_provider(ParamProvider(param_type=DICFKey))
server.append_provider(ParamProvider(param_type=SigmaDICFKey))
server.append_provider(ParamProvider(param_type=GrottoDICFKey))  # EQUAL
server.append_provider(ParamProvider(param_type=B2AKey))  # B2A
server.set_nonlinear_operation_provider()
server.online()

# test_sizes = [400, 500]
test_sizes = [100, 200, 300, 400, 500]
# test_sizes = [300]
warmup_times = 2
run_times = 3

server.send(test_sizes)
server.send(warmup_times)
server.send(run_times)


class TestServer(unittest.TestCase):
    results = None

    def __init__(self, methodName='runTest', data_amount=0):
        super(TestServer, self).__init__(methodName)
        self.data_amount = data_amount

    @classmethod
    def setUpClass(cls):
        cls.results = []

    def setUp(self):
        self.x = torch.rand([self.data_amount, self.data_amount]).to(DEVICE)
        self.y = torch.rand([self.data_amount, self.data_amount]).to(DEVICE)

        x_0, x_1 = ArithmeticSecretSharing.share(RingTensor.convert_to_ring(self.x), 2)
        server.send(x_1)
        self.share_x = x_0
        x_0.party = server

        y_0, y_1 = ArithmeticSecretSharing.share(RingTensor.convert_to_ring(self.y), 2)
        server.send(y_1)
        self.share_y = y_0
        y_0.party = server

    @classmethod
    def tearDownClass(cls):
        file_name = f'{BIT_LEN}_{DEVICE}_{DEBUG_LEVEL}_test_results.csv'
        file_exists = os.path.isfile(file_name)
        with open(file_name, 'a', newline='') as csvfile:
            fieldnames = ['Test Method'] + [f'{amt}' for amt in test_sizes]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            method_results = {}
            for result in cls.results:
                method = result['Test Method']
                amount = result['Test Data Amount']
                value = result['Result Value']
                if method not in method_results:
                    method_results[method] = {f'{amt}': '' for amt in test_sizes}
                method_results[method][f'{amount}'] = value

            for method, values in method_results.items():
                row = {'Test Method': method}
                row.update(values)
                writer.writerow(row)

    def tearDown(self):
        self.__class__.results.append({
            'Test Method': self._testMethodName,
            'Test Data Amount': self.data_amount,
            'Result Value': self.result
        })

    def run_test(self, test_func):
        sum = 0
        for i in range(warmup_times):
            test_func()
        for i in range(run_times):
            start_time = time.perf_counter()
            test_func()
            end_time = time.perf_counter()
            sum += end_time - start_time
            torch.cuda.empty_cache()
        self.result = sum / run_times
        print(f"Test {self._testMethodName} with data amount {self.data_amount} took {self.result} seconds")

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
            triples = MatmulTriples.gen(10, self.share_x.shape, self.share_y.shape)
            server.providers['MatmulTriples'].param = triples[0]
            server.send(triples[1])
            server.providers['MatmulTriples'].load_mat_beaver()

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

    # def test_12_inv_sqrt(self):
    #     print("======================rsqrt(y)=========================")
    #     self.run_test(lambda: ArithmeticSecretSharing.rsqrt(self.share_y))


def parametrize(testcase_klass, data_amounts):
    test_loader = unittest.TestLoader()
    test_names = test_loader.getTestCaseNames(testcase_klass)
    suite = unittest.TestSuite()
    for name in test_names:
        for data_amount in data_amounts:
            suite.addTest(testcase_klass(name, data_amount))
    return suite


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTests(parametrize(TestServer, test_sizes))
    return suite


if __name__ == '__main__':
    unittest.main()
