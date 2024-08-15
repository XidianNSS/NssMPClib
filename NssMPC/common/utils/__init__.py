from NssMPC.common.utils.cuda_utils import cuda_matmul
from NssMPC.common.utils.debug_utils import *


def convert_tensor(tensor):
    res = tensor[..., 0].unsqueeze(-1)
    return res
