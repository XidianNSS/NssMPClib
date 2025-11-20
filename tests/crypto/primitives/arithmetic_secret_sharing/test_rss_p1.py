from NssMPC.infra.mpc.party import Party3PC
from NssMPC.runtime import PartyRuntime, HONEST_MAJORITY, SEMI_HONEST

ns = [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]

client = Party3PC(1, SEMI_HONEST)
client.online()


def test_multiplication(share_x, share_y):
    share_z = share_x * share_y
    for _ in range(100):
        # print(f"迭代测试乘法...{_}")
        share_z = share_x * share_y
    res_share_z = share_z.restore().convert_to_real_field()


def test_greater_equal(share_x, share_y):
    share_z = share_x >= share_y
    for _ in range(100):
        share_z = share_x >= share_y
    res_share_z = share_z.restore().convert_to_real_field()


if __name__ == "__main__":

    for n in ns:
        with PartyRuntime(client):
            # 准备测试数据
            share_x = client.recv(0)
            share_y = client.recv(0)

            test_multiplication(share_x, share_y)

            test_greater_equal(share_x, share_y)
