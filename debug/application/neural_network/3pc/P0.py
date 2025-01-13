import time

import torch.utils.data

import NssMPC.application.neural_network as nn
from NssMPC import RingTensor
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import share
from data.AlexNet.Alexnet import AlexNet


secure_model = 0
if secure_model == 0:
    from NssMPC.application.neural_network.party.neural_network_party import HonestMajorityNeuralNetWork3PC as Party
else:
    from NssMPC.application.neural_network.party.neural_network_party import SemiHonest3PCParty as Party

if __name__ == '__main__':
    P = Party(0)
    P.set_multiplication_provider()
    P.set_trunc_provider()
    P.set_comparison_provider()
    P.online()

    test_input = torch.randint(-10, 10, [16, 1, 28, 28]) * 1.0
    print("test_input:", test_input)
    net = AlexNet()
    test_output = net(test_input)
    print("test_output", test_output)

    print("开始分享权重")
    shared_param = nn.utils.share_model(net, share_type=32)
    local_param = shared_param[0]
    P1_param = shared_param[1]
    P2_param = shared_param[2]
    P.send(1, P1_param)
    P.send(2, P2_param)

    print("预处理一些东西")
    num = P.dummy_model(net, test_input)
    net = nn.utils.load_model(net, local_param)

    print("开始分享输入")

    share_input = share(RingTensor.convert_to_ring(test_input), P)
    print("share input", share_input.restore().convert_to_real_field())
    st = time.time()
    output = net(share_input)
    et = time.time()
    print("time cost", et - st)
    print("output", output.restore().convert_to_real_field())
