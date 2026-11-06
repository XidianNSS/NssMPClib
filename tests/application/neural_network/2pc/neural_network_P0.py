import argparse
import os

import torch

import nssmpc.application.neural_network as nn
from data.AlexNet.Alexnet import AlexNet
from nssmpc import SEMI_HONEST, Party2PC, PartyRuntime
from nssmpc.config import DEBUG_LEVEL, NN_path
from tests.application.neural_network.smoke_model import SmokeNet
from tests.parameter_setup import prepare_model_triples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-party inference with test preprocessing.')
    parser.add_argument('--smoke', action='store_true', help='Use a small model without pretrained weights.')
    parser.add_argument('--num-samples', type=int, default=1024, help='Number of CIFAR10 test examples.')
    args = parser.parse_args()
    if not 1 <= args.num_samples <= 10000:
        parser.error('--num-samples must be between 1 and 10000')

    party = Party2PC(0, SEMI_HONEST, master_port=os.environ.get('MASTER_PORT', '29500'))
    try:
        party.online()
        with PartyRuntime(party):
            model_class = SmokeNet if args.smoke else AlexNet
            plaintext_model = model_class().eval()
            if args.smoke:
                with torch.no_grad():
                    for parameter in plaintext_model.parameters():
                        parameter.fill_(0.125)
            else:
                plaintext_model.load_state_dict(torch.load(NN_path + 'AlexNet_CIFAR10.pkl'))
            shape = (1, 1, 4, 4) if args.smoke else (1, 3, 32, 32)
            count = 1 if args.smoke or DEBUG_LEVEL else args.num_samples
            prepare_model_triples(party, plaintext_model, torch.zeros(shape), count)

            shared_param = nn.utils.share_model_param(model=plaintext_model)
            SecAlexNet = nn.utils.convert_model(model_class)
            ciphertext_model = SecAlexNet()
            ciphertext_model = nn.utils.load_shared_param(ciphertext_model, shared_param)
            shared_data_loader = nn.utils.SharedDataLoader(src_id=1)

            for data in shared_data_loader:
                secret_result = ciphertext_model(data)
                secret_result.recon(target_id=1)
    finally:
        party.close()
