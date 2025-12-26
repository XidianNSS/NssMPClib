import torch

import nssmpc.application.neural_network as nn
from nssmpc import PartyRuntime, Party2PC, SEMI_HONEST
from nssmpc.config import NN_path
from data.AlexNet.Alexnet import AlexNet

if __name__ == '__main__':

    party = Party2PC(0, SEMI_HONEST)
    party.online()
    with PartyRuntime(party):

        plaintext_model = AlexNet()
        plaintext_model.load_state_dict(torch.load(NN_path + 'AlexNet_CIFAR10.pkl'))

        shared_param = nn.utils.share_model_param(model=plaintext_model)
        SecAlexNet = nn.utils.convert_model(AlexNet)
        ciphertext_model = SecAlexNet()
        ciphertext_model = nn.utils.load_shared_param(ciphertext_model, shared_param)
        shared_data_loader = nn.utils.SharedDataLoader(src_id=1)

        for data in shared_data_loader:
            secret_result = ciphertext_model(data)
            secret_result.recon(target_id=1)

    party.close()
