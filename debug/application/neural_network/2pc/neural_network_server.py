import torch

import NssMPC.application.neural_network as nn
from NssMPC.config.runtime import PartyRuntime
from data.AlexNet.Alexnet import AlexNet
from NssMPC.config import NN_path

# set server and client address

if __name__ == '__main__':
    server = nn.party.NeuralNetworkCS(type='server')
    server.online()
    with PartyRuntime(server):
        net = AlexNet()
        net.load_state_dict(torch.load(NN_path + 'AlexNet_MNIST.pkl'))
        shared_param, shared_param_for_other = nn.utils.share_model(net)
        server.send(shared_param_for_other)

        num = server.dummy_model(net)

        net = nn.utils.load_model(net, shared_param)

        while num:
            shared_data = server.receive()
            server.inference(net, shared_data)
            num -= 1

    server.close()
