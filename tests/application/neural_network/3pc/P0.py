import argparse
import os

import torch.utils.data

import nssmpc.application.neural_network as nn
from data.AlexNet.Alexnet import AlexNet
from nssmpc import HONEST_MAJORITY, Party3PC, PartyRuntime, SecretTensor
from nssmpc.infra.utils.profiling import RuntimeTimer
from tests.application.neural_network.smoke_model import SmokeNet
from tests.parameter_setup import prepare_model_triples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Honest-majority inference with test preprocessing.')
    parser.add_argument('--smoke', action='store_true', help='Use a small model and synthetic input.')
    args = parser.parse_args()
    P = Party3PC(0, HONEST_MAJORITY, master_port=os.environ.get('MASTER_PORT', '29500'))
    try:
        P.online()
        with PartyRuntime(P):
            model_class = SmokeNet if args.smoke else AlexNet
            test_input = (
                torch.arange(16).reshape(1, 1, 4, 4).float() / 16
                if args.smoke
                else torch.randint(-10, 10, [1, 3, 32, 32]) * 1.0
            )
            print("test_input:", test_input)
            plaintext_model = model_class().eval()
            if args.smoke:
                with torch.no_grad():
                    for parameter in plaintext_model.parameters():
                        parameter.fill_(0.125)
            prepare_model_triples(P, plaintext_model, test_input)
            test_output = plaintext_model(test_input)
            print("test_output", test_output)
            # Share model parameters
            shared_param = nn.utils.share_model_param(model=plaintext_model)
            # Convert to secure model class
            SecAlexNet = nn.utils.convert_model(model_class)
            # Instantiate secure model
            ciphertext_model = SecAlexNet()
            # Load shared parameters
            net = nn.utils.load_shared_param(ciphertext_model, shared_param)
            # Share input data
            share_input = SecretTensor(tensor=test_input)
            # Inference and profiling
            with RuntimeTimer(enable_comm_stats=True):
                output = net(share_input)
            # Reconstruct output to Party 0 and print
            result = output.recon(target_id=0).convert_to_real_field()
            P.communicator.barrier()
            print("output", result)
            if args.smoke:
                torch.testing.assert_close(result.cpu(), test_output, atol=0.04, rtol=0.02)
    finally:
        P.close()
