import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from NssMPC.application.neural_network.party.neural_network_party import HonestMajorityNeuralNetWork3PC
import torch
import NssMPC.application.neural_network as nn
from data.AlexNet.Alexnet import AlexNet
from NssMPC.config import NN_path

# 测试恶意乘法


if __name__ == '__main__':
    Party = HonestMajorityNeuralNetWork3PC(0)
    Party.online()

    print("开始分享权重")
    net = AlexNet()
    net.load_state_dict(torch.load(NN_path + 'AlexNet_MNIST.pkl'))
    shared_param = nn.utils.share_model(net, share_type=32)
    local_param = shared_param[0]
    P1_param = shared_param[1]
    P2_param = shared_param[2]
    Party.send(1, P1_param)
    Party.send(2, P2_param)
    transform1 = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.MNIST(root=NN_path, train=False, download=True, transform=transform1)
    indices = list(range(16))
    subset_data = Subset(test_set, indices)
    test_loader = torch.utils.data.DataLoader(subset_data, batch_size=8, shuffle=False, num_workers=0)

    print("预处理一些东西")
    num = Party.dummy_model(net, test_loader)
    net = nn.utils.load_model(net, local_param)

    print("开始分享输入")
