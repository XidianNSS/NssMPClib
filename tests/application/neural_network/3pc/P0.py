import torch.utils.data

import nssmpc.application.neural_network as nn
from nssmpc import Party3PC, SEMI_HONEST, PartyRuntime, SecretTensor, HONEST_MAJORITY
from nssmpc.infra.utils.profiling import RuntimeTimer
from data.AlexNet.Alexnet import AlexNet

if __name__ == '__main__':
    # P = Party3PC(0, SEMI_HONEST)
    P = Party3PC(0, HONEST_MAJORITY)
    P.online()
    with PartyRuntime(P):
        test_input = torch.randint(-10, 10, [1, 3, 32, 32]) * 1.0
        print("test_input:", test_input)
        plaintext_model = AlexNet()
        test_output = plaintext_model(test_input)
        print("test_output", test_output)
        # Share model parameters
        shared_param = nn.utils.share_model_param(model=plaintext_model)
        # Convert to secure model class
        SecAlexNet = nn.utils.convert_model(AlexNet)
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
        print("output", output.recon(target_id=0).convert_to_real_field())
    P.close()
