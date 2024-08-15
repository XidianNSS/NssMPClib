import torch
from NssMPC.application.neural_network.functional.functional import torch2share
from NssMPC.config import DEVICE


class SecLinear(torch.nn.Module):
    def __init__(self, input, output, weight=None, bias=None, device=DEVICE):
        super(SecLinear, self).__init__()
        self.device = device
        self.weight = torch.nn.Parameter(torch.zeros([output, input], dtype=torch.int64), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros([output], dtype=torch.int64), requires_grad=False)

    def forward(self, x):
        weight = torch2share(self.weight, x.__class__, x.dtype, x.party).T
        bias = torch2share(self.bias, x.__class__, x.dtype, x.party)
        z = (x @ weight) + bias
        return z
