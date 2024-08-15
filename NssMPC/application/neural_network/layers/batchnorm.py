import torch
from NssMPC.application.neural_network.functional.functional import torch2share
from NssMPC.config import data_type


class SecBatchNorm2d(torch.nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True,
                 track_running_stats: bool = True, device=None):
        super(SecBatchNorm2d, self).__init__()
        self.weight = torch.nn.Parameter(torch.zeros([num_features], dtype=data_type), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros([num_features], dtype=data_type), requires_grad=False)

    def forward(self, x):
        epsilon1 = torch2share(self.weight, x.__class__, x.dtype, x.party).unsqueeze(1).unsqueeze(2)
        epsilon2 = torch2share(self.bias, x.__class__, x.dtype, x.party).unsqueeze(1).unsqueeze(2)
        return (x * epsilon1) + epsilon2
