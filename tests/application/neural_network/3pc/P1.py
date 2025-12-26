import nssmpc.application.neural_network as nn
from nssmpc import Party3PC, SEMI_HONEST, PartyRuntime, SecretTensor, HONEST_MAJORITY
from data.AlexNet.Alexnet import AlexNet

if __name__ == '__main__':
    # P = Party3PC(1, SEMI_HONEST)
    P = Party3PC(1, HONEST_MAJORITY)
    P.online()
    with PartyRuntime(P):
        # Receive weights
        local_param = nn.utils.share_model_param(src_id=0)
        # Convert to secure model class
        SecAlexNet = nn.utils.convert_model(AlexNet)
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
    P.close()
