import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import NssMPC.application.neural_network as nn
from NssMPC.common.utils import get_time
from NssMPC.config import DEVICE, NN_path
from NssMPC.config.runtime import PartyRuntime
from data.AlexNet.Alexnet import AlexNet

# set server and client address

if __name__ == '__main__':
    client = nn.party.NeuralNetworkCS(type='client')
    client.online()
    with PartyRuntime(client):
        transform1 = transforms.Compose([transforms.ToTensor()])
        test_set = torchvision.datasets.CIFAR10(root=NN_path, train=False, download=True, transform=transform1)

        indices = list(range(1024))
        subset_data = Subset(test_set, indices)
        test_loader = torch.utils.data.DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=0)

        net = AlexNet()

        shared_param = client.receive()

        num = client.dummy_model(test_loader)

        net = nn.utils.load_model(net, shared_param)
        correct_total = 0
        total_total = 0

        for data in test_loader:
            correct = 0
            total = 0
            images, labels = data

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            shared_data, shared_data_for_other = nn.utils.share_data(images)
            client.send(shared_data_for_other)

            res = get_time(client.inference, net, shared_data)

            _, predicted = torch.max(res, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            total_total += total
            correct_total += correct

            print('Accuracy of the network on test images:{}%'.format(100 * correct / total))

        print('Accuracy of the network on test images:{}%'.format(100 * correct_total / total_total))

    client.close()
