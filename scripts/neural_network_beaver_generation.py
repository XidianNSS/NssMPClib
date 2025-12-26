import torch

from nssmpc import Party2PC, SEMI_HONEST, Party3PC, HONEST_MAJORITY
from nssmpc.application.neural_network.utils.beaver import gen_mat_beaver
from data.AlexNet.Alexnet import AlexNet

dummy_input = torch.randn(1, 3, 32, 32)
model = AlexNet()
party = Party2PC(0, SEMI_HONEST)
# party = Party3PC(0, HONEST_MAJORITY)

gen_mat_beaver(dummy_input, model, 10, party, save=True)
