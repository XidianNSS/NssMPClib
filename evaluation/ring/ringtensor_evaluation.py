"""
ringtensor_evaluation.py 用于对RingTensor的基础性能进行测试于评估
对于每一个待测试的内容，去除开头第一次，一共执行5次取平均
"""
import time
import torch
import unittest
from NssMPC.common.ring import RingTensor
from NssMPC.config import DEVICE

# 定义测试规模
num = 12800000
print("测试数据规模: ", num)

image_size = (32, 32, 3, 3)
k_size = 3
stride = 3

"""
convert_to_ring
"""
t = torch.rand([num], device=DEVICE)
r = RingTensor.convert_to_ring(t)


class TestRingTensor(unittest.TestCase):
    def test_01_init(self):
        """
        从torch.tensor初始化
        """
        start_time = time.time()
        _ = RingTensor(t)
        end_time = time.time()
        print("从torch.tensor初始化所消耗的时间: ", end_time - start_time)

    def test_02_convert_to_ring(self):
        """
        convert_to_ring
        """
        start_time = time.time()
        _ = RingTensor.convert_to_ring(t)
        end_time = time.time()
        print("convert_to_ring所消耗的时间: ", end_time - start_time)

    def test_03_get_item(self):
        """
        get item 测试
        """
        start_time = time.time()
        _ = r[num // 2]
        end_time = time.time()
        print("get item 测试: ", end_time - start_time)

    def test_04_set_item(self):
        """
        set item 测试
        """
        k = RingTensor(0)
        start_time = time.time()
        r[num // 2] = k
        end_time = time.time()
        print("get item 测试: ", end_time - start_time)

    def test_05_add(self):
        """
        add 测试
        """
        start_time = time.time()
        _ = r + r
        end_time = time.time()
        print("add 测试 (RingTensor): ", end_time - start_time)

        start_time = time.time()
        _ = r + t
        end_time = time.time()
        print("add 测试 (torch.tensor): ", end_time - start_time)

    def test_06_mul(self):
        start_time = time.time()
        _ = r * r
        end_time = time.time()
        print("mul 测试 (RingTensor): ", end_time - start_time)

        m = r[0]
        start_time = time.time()
        _ = r * m
        end_time = time.time()
        print("mul 测试 (single RingTensor): ", end_time - start_time)
        start_time = time.time()

        _ = r * 3
        end_time = time.time()
        print("mul 测试 ( int): ", end_time - start_time)

    def test_07_mod(self):
        start_time = time.time()
        _ = r % 2
        end_time = time.time()
        print("mod 测试: ", end_time - start_time)

    def test_08_mat_mul(self):
        m = r.T
        start_time = time.time()
        _ = r @ m
        end_time = time.time()
        print("mat mul 测试: ", end_time - start_time)

    def test_09_ture_div(self):
        m = r[0]
        start_time = time.time()
        _ = r / m
        end_time = time.time()
        print("true div 测试: ", end_time - start_time)

    def test_10_floor_div(self):
        m = r[0]
        start_time = time.time()
        _ = r // m
        end_time = time.time()
        print("floor div 测试: ", end_time - start_time)

    def test_11_eq(self):
        start_time = time.time()
        _ = r == r
        end_time = time.time()
        print("eq 测试: ", end_time - start_time)

    def test_12_ge(self):
        start_time = time.time()
        _ = r >= r
        end_time = time.time()
        print("ge 测试: ", end_time - start_time)

    def test_13_or(self):
        start_time = time.time()
        _ = r | r
        end_time = time.time()
        print("or 测试: ", end_time - start_time)

    def test_14_and(self):
        start_time = time.time()
        _ = r & r
        end_time = time.time()
        print("and 测试: ", end_time - start_time)

    def test_15_save(self):
        start_time = time.time()
        r.save("tmp.pth")
        end_time = time.time()
        print("save 测试: ", end_time - start_time)

    def test_16_load(self):
        start_time = time.time()
        _ = RingTensor.load_from_file("tmp.pth")
        end_time = time.time()
        print("load 测试: ", end_time - start_time)

    def test_17_to_cuda(self):
        r_cpu = r.clone().to('cpu')
        start_time = time.time()
        r_cpu.to('cuda')
        end_time = time.time()
        print("to cuda 测试: ", end_time - start_time)

    def test_18_to_cpu(self):
        r_cuda = r.clone().to('cuda')
        start_time = time.time()
        r_cuda.to('cpu')
        end_time = time.time()
        print("to cpu 测试: ", end_time - start_time)

    def test_19_get_bit(self):
        start_time = time.time()
        _ = r.get_bit(8)
        end_time = time.time()
        print("get bit 测试: ", end_time - start_time)

    def test_20_bit_slice(self):
        start_time = time.time()
        _ = r.bit_slice(3, 8)
        end_time = time.time()
        print("get bit 测试: ", end_time - start_time)

    def test_21_img2col(self):
        image = torch.rand(image_size)
        img = RingTensor.convert_to_ring(image)
        start_time = time.time()
        _ = RingTensor.img2col(img, k_size, stride)
        end_time = time.time()
        print("img2col 测试: ", end_time - start_time)
