import torch


class SecDropout(torch.nn.Module):

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super(SecDropout, self).__init__()
        pass

    def forward(self, x):
        return x


def _SecDropout(x):
    return x
