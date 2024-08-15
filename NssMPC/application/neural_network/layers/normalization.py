import torch
from NssMPC.application.neural_network.functional.functional import torch2share
from NssMPC.config import data_type


class SecLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super(SecLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones([normalized_shape], dtype=data_type), requires_grad=False)
        self.bias = torch.nn.Parameter(
            torch.zeros([normalized_shape], dtype=data_type), requires_grad=False)
        self.scale = None
        self.zero_point = None
        self.elementwise_affine = elementwise_affine

    def forward(self, x):
        mean = x.sum(dim=-1).unsqueeze(-1) / self.normalized_shape

        z = x - mean
        inv_sqrt_variance = x.__class__.rsqrt(((z * z).sum(dim=-1).unsqueeze(-1)) / self.normalized_shape)

        weight = torch2share(self.weight, x.__class__, x.dtype, x.party)
        bias = torch2share(self.bias, x.__class__, x.dtype, x.party)

        z = z * inv_sqrt_variance * weight + bias

        return z
