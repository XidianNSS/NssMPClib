import csv
import time
from datetime import datetime
from pathlib import Path

import torch

import NssMPC
from NssMPC import PartyRuntime, SEMI_HONEST, Party3PC
from NssMPC.config import DEVICE
from NssMPC.infra.utils.debug_utils import bytes_convert

server = Party3PC(0, SEMI_HONEST)

ns = [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]
original_path = Path("./app/nssmpclib/tmp/output.csv")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 使用 .with_name() 构造新文件名
output_file = original_path.with_name(
    f"{original_path.stem}_{timestamp}{original_path.suffix}"
)
server.online()


def prepare_data(n):
    x = torch.rand(n).to(DEVICE)
    y = torch.rand(n).to(DEVICE)

    share_x = NssMPC.SecretTensor(tensor=x)
    share_y = NssMPC.SecretTensor(tensor=y)

    return share_x, share_y, x, y


def test_multiplication(share_x, share_y, x, y):
    print("===================x * y==========================")
    communicator = server.communicator
    before_comm_rounds = communicator.comm_stats['send_count']
    before_comm_bytes = communicator.comm_stats['send_bytes']
    share_z = share_x * share_y
    now_comm_rounds = communicator.comm_stats['send_count']
    now_comm_bytes = communicator.comm_stats['send_bytes']

    time1 = time.time()
    for _ in range(100):
        # print(f"迭代测试乘法...{_}")
        share_z = share_x * share_y
    time2 = time.time()
    res_share_z = share_z.restore().convert_to_real_field()
    comparision_result = torch.allclose((x * y).to(res_share_z),
                                        res_share_z, atol=1e-2, rtol=1e-2)
    print("结果对比: ", comparision_result)
    return (time2 - time1) / 100, now_comm_rounds - before_comm_rounds, now_comm_bytes - before_comm_bytes


def test_greater_equal(share_x, share_y, x, y):
    print("===================x >= y=========================")
    communicator = server.communicator
    before_comm_rounds = communicator.comm_stats['send_count']
    before_comm_bytes = communicator.comm_stats['send_bytes']
    share_z = share_x >= share_y
    now_comm_rounds = communicator.comm_stats['send_count']
    now_comm_bytes = communicator.comm_stats['send_bytes']

    time1 = time.time()
    for _ in range(100):
        share_z = share_x >= share_y
    time2 = time.time()
    res_share_z = share_z.restore().convert_to_real_field()
    comparision_result = res_share_z
    print("结果对比: ", comparision_result)
    return (time2 - time1) / 100, now_comm_rounds - before_comm_rounds, now_comm_bytes - before_comm_bytes


if __name__ == "__main__":
    all_rows = []
    all_rows.append(["operation", "n", "time_cost",
                     "comm_rounds", "comm_bytes"])
    for n in ns:
        with PartyRuntime(server):
            # 准备测试数据
            share_x, share_y, x, y = prepare_data(n)

            time_cost, rouds, comm_bytes = test_multiplication(
                share_x, share_y, x, y)
            row = ["multiplication", n, time_cost, rouds, bytes_convert(comm_bytes)]
            all_rows.append(row)

            time_cost, rouds, comm_bytes = test_greater_equal(
                share_x, share_y, x, y)
            row = ["comparision", n, time_cost, rouds, bytes_convert(comm_bytes)]
            all_rows.append(row)
            time.sleep(1)
    with open(output_file, "a") as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)
