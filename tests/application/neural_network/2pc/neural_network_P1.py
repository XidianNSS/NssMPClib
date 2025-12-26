import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

import nssmpc.application.neural_network as nn
from nssmpc import PartyRuntime, SEMI_HONEST, Party2PC
from nssmpc.config import NN_path
from nssmpc.infra.utils.profiling import RuntimeTimer
from data.AlexNet.Alexnet import AlexNet

if __name__ == '__main__':
    party = Party2PC(1, SEMI_HONEST)
    party.online()

    with PartyRuntime(party):
        transform1 = transforms.Compose([transforms.ToTensor()])
        test_set = torchvision.datasets.CIFAR10(root=NN_path, train=False, download=True, transform=transform1)

        indices = list(range(1024))
        subset_data = Subset(test_set, indices)
        test_loader = torch.utils.data.DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=0)

        shared_param = nn.utils.share_model_param(src_id=0)
        SecAlexNet = nn.utils.convert_model(AlexNet)
        ciphertext_model = SecAlexNet()
        ciphertext_model = nn.utils.load_shared_param(ciphertext_model, shared_param)
        shared_data_loader = nn.utils.SharedDataLoader(data_loader=test_loader)

        correct_total = 0
        total_total = 0

        for data in shared_data_loader:
            correct = 0
            total = 0
            inputs, labels = data

            with RuntimeTimer(tag="Inference", enable_comm_stats=True):
                secret_result = ciphertext_model(inputs)

            plaintext_result = secret_result.recon(target_id=1).convert_to_real_field()

            _, predicted = torch.max(plaintext_result, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            total_total += total
            correct_total += correct

            print('Accuracy of the communication on test images:{}%'.format(100 * correct / total))

        print('Accuracy of the communication on test images:{}%'.format(100 * correct_total / total_total))

    party.close()
