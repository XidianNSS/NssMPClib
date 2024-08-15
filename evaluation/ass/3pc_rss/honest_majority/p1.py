import unittest
import time
from NssMPC import RingTensor
from NssMPC.crypto.aux_parameter import RssMatmulTriples
from NssMPC.secure_model.mpc_party.honest_majority import HonestMajorityParty
from NssMPC.crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
from NssMPC.config import DEBUG_LEVEL
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import *

# 定义测试规模
num = 100
print("测试数据规模: ", num, "*", num)

id = 1
Party = HonestMajorityParty(id=id)
Party.set_comparison_provider()
Party.online()

x = ReplicatedSecretSharing.random([num, num], party=Party)
y = ReplicatedSecretSharing.random([num, num], party=Party)

triples = RssMatmulTriples.gen(1, x.shape, y.shape)

if DEBUG_LEVEL != 2:  # TODO: DEBUG_LEVEL统一
    triple = Party.receive(0)
    Party.providers[RssMatmulTriples.__name__].param = [triple]
    Party.providers[RssMatmulTriples.__name__].load_mat_beaver()


class TestRssMalicious(unittest.TestCase):
    def test_01_get_item(self):
        start_time = time.time()
        _ = x[num // 2]
        end_time = time.time()
        print("get item time: ", end_time - start_time)

    def test_02_set_item(self):
        r = ReplicatedSecretSharing.random([1], party=Party)
        start_time = time.time()
        x[num // 2][num // 2] = r
        end_time = time.time()
        print("set item time: ", end_time - start_time)

    def test_03_add(self):
        start_time = time.time()
        _ = x + y
        end_time = time.time()
        print("add time: ", end_time - start_time)

    def test_04_mul(self):
        start_time = time.time()
        _ = x * y
        end_time = time.time()
        print("mul time: ", end_time - start_time)

    def test_04_mat_mul(self):
        start_time = time.time()
        _ = x @ y
        end_time = time.time()
        print("mat mul time: ", end_time - start_time)

    # def test_05_ge(self):
    #     start_time = time.time()
    #     _ = x >= y
    #     end_time = time.time()
    #     print("ge time: ", end_time - start_time)

    def test_06_open(self):
        start_time = time.time()
        _ = open(x)
        end_time = time.time()
        print("open time: ", end_time - start_time)

    def test_07_coin(self):
        start_time = time.time()
        _ = coin([num, num], party=Party)
        end_time = time.time()
        print("coin time: ", end_time - start_time)

    def test_08_recon(self):
        start_time = time.time()
        _ = recon(x, 0)
        end_time = time.time()
        print("recon time: ", end_time - start_time)

    def test_09_share(self):
        if Party.party_id == 0:
            r = RingTensor.random([num, num])
            start_time = time.time()
            share(r, Party)
            end_time = time.time()
            print("share time: ", end_time - start_time)

        else:
            start_time = time.time()
            _ = receive_share_from(0, Party)
            end_time = time.time()
            print("share time: ", end_time - start_time)

    def test_10_check_zero(self):
        z = ReplicatedSecretSharing.zeros([num * num], party=Party)
        start_time = time.time()
        _ = check_zero(z)
        end_time = time.time()
        print("check zero time: ", end_time - start_time)
