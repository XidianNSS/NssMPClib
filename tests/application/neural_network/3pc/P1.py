import argparse
import os

import nssmpc.application.neural_network as nn
from data.AlexNet.Alexnet import AlexNet
from nssmpc import HONEST_MAJORITY, Party3PC, PartyRuntime, SecretTensor
from tests.application.neural_network.smoke_model import SmokeNet
from tests.parameter_setup import prepare_model_triples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Honest-majority inference with test preprocessing.')
    parser.add_argument('--smoke', action='store_true', help='Use a small model and synthetic input.')
    args = parser.parse_args()
    P = Party3PC(1, HONEST_MAJORITY, master_port=os.environ.get('MASTER_PORT', '29500'))
    try:
        P.online()
        with PartyRuntime(P):
            prepare_model_triples(P)
            # Receive weights
            local_param = nn.utils.share_model_param(src_id=0)
            # Convert to secure model class
            SecAlexNet = nn.utils.convert_model(SmokeNet if args.smoke else AlexNet)
            # Instantiate secure model
            ciphertext_model = SecAlexNet()
            # Load weights
            ciphertext_model = nn.utils.load_shared_param(ciphertext_model, local_param)
            # Receive input
            share_input = SecretTensor(src_id=0)
            # Inference
            output = ciphertext_model(share_input)
            # Reconstruct output to Party 0
            output.recon(target_id=0)
            P.communicator.barrier()
    finally:
        P.close()
