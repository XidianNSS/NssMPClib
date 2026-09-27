import argparse
import os

import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

import nssmpc.application.neural_network as nn
from data.AlexNet.Alexnet import AlexNet
from nssmpc import SEMI_HONEST, Party2PC, PartyRuntime
from nssmpc.config import NN_path
from nssmpc.infra.utils.profiling import RuntimeTimer
from tests.application.neural_network.smoke_model import SmokeNet
from tests.parameter_setup import prepare_model_triples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-party inference with test preprocessing.')
    parser.add_argument('--smoke', action='store_true', help='Use a small model without downloading CIFAR10.')
    parser.add_argument('--num-samples', type=int, default=1024, help='Number of CIFAR10 test examples.')
    args = parser.parse_args()
    if not 1 <= args.num_samples <= 10000:
        parser.error('--num-samples must be between 1 and 10000')
    party = Party2PC(1, SEMI_HONEST, master_port=os.environ.get('MASTER_PORT', '29500'))
    try:
        party.online()

        with PartyRuntime(party):
            prepare_model_triples(party)
            transform1 = transforms.Compose([transforms.ToTensor()])
            if args.smoke:
                smoke_input = torch.arange(16).reshape(1, 1, 4, 4).float() / 16
                subset_data = torch.utils.data.TensorDataset(smoke_input, torch.zeros(1, dtype=torch.long))
                reference_model = SmokeNet().eval()
                with torch.no_grad():
                    for parameter in reference_model.parameters():
                        parameter.fill_(0.125)
                    expected = reference_model(smoke_input)
            else:
                test_set = torchvision.datasets.CIFAR10(root=NN_path, train=False, download=True, transform=transform1)
                subset_data = Subset(test_set, list(range(args.num_samples)))
            test_loader = torch.utils.data.DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=0)

            shared_param = nn.utils.share_model_param(src_id=0)
            SecAlexNet = nn.utils.convert_model(SmokeNet if args.smoke else AlexNet)
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
                if args.smoke:
                    torch.testing.assert_close(plaintext_result.cpu(), expected, atol=0.04, rtol=0.02)

                _, predicted = torch.max(plaintext_result, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                total_total += total
                correct_total += correct

                print('Accuracy of the communication on test images:{}%'.format(100 * correct / total))

            print('Accuracy of the communication on test images:{}%'.format(100 * correct_total / total_total))
    finally:
        party.close()
