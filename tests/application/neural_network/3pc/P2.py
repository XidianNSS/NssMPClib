import NssMPC.application.neural_network as nn
from NssMPC.application.neural_network.party import PartyNeuralNetWork3PC
from NssMPC.protocols.honest_majority_3pc.base import receive_share_from
from NssMPC.runtime import PartyRuntime, SEMI_HONEST
from data.AlexNet.Alexnet import AlexNet

if __name__ == '__main__':
    P = PartyNeuralNetWork3PC(2, SEMI_HONEST)
    P.online()

    with PartyRuntime(P):
        net = AlexNet()
        print("接收权重")
        local_param = P.recv(0)

        print("预处理一些东西")
        num = P.dummy_model()
        net = nn.utils.load_model(net, local_param)

        print("接收输入")

        share_input = receive_share_from(0, P)
        print("share input", share_input.restore().convert_to_real_field())
        for i in range(10):
            output = net(share_input)
        print("output", output.restore().convert_to_real_field())
