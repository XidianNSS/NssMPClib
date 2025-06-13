import NssMPC.application.neural_network as nn
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import *
from data.AlexNet.Alexnet import AlexNet

# 测试恶意乘法
secure_model = 0

if secure_model == 0:
    from NssMPC.application.neural_network.party.neural_network_party import HonestMajorityNeuralNetWork3PC as Party
else:
    from NssMPC.application.neural_network.party.neural_network_party import SemiHonest3PCParty as Party


# 测试恶意乘法
if __name__ == '__main__':
    P = Party(1)
    P.set_multiplication_provider()
    P.set_trunc_provider()
    P.set_comparison_provider()
    P.online()
    net = AlexNet()
    print("接收权重")
    local_param = P.receive(0)

    print("预处理一些东西")
    num = P.dummy_model()
    net = nn.utils.load_model(net, local_param)
    print("接收输入")

    share_input = receive_share_from(0, P)
    print("share input", share_input.restore().convert_to_real_field())
    output = net(share_input)
    print("output", output.restore().convert_to_real_field())
