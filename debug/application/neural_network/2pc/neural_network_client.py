import time

import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

from application.neural_network.model.model_converter import load_secure_model, share_data
from application.neural_network.party.neural_network_party import NeuralNetworkCS
from common.utils import get_time
from config.base_configs import DEVICE
from config.network_configs import *
from data.neural_network.AlexNet.Alexnet import AlexNet

# set server and client address
server_server_address = (SERVER_IP, SERVER_SERVER_PORT)
server_client_address = (SERVER_IP, SERVER_CLIENT_PORT)

client_server_address = (CLIENT_IP, CLIENT_SERVER_PORT)
client_client_address = (CLIENT_IP, CLIENT_CLIENT_PORT)

if __name__ == '__main__':

    client = NeuralNetworkCS(type='client')
    client.connect(client_server_address, client_client_address, server_server_address, server_client_address)

    transform1 = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.MNIST(root='data/neural_network/', train=False, download=True, transform=transform1)

    indices = list(range(5))
    subset_data = Subset(test_set, indices)
    test_loader = torch.utils.data.DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=0)

    net = AlexNet()

    shared_param = client.receive()

    num = client.dummy_model(test_loader)

    net = load_secure_model(net, shared_param)

    correct_total = 0
    total_total = 0

    for data in test_loader:
        correct = 0
        total = 0
        images, labels = data

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        shared_data, shared_data_for_other = share_data(images)
        client.send(shared_data_for_other)

        res = get_time(client.inference, net, shared_data)

        _, predicted = torch.max(res, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        total_total += total
        correct_total += correct

        print('Accuracy of the network on test images:{}%'.format(100 * correct / total))

    print('Accuracy of the network on test images:{}%'.format(100 * correct_total / total_total))
    time.sleep(2)
    client.close()
