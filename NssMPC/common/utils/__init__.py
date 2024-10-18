#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.common.utils.cuda_utils import cuda_matmul
from NssMPC.common.utils.debug_utils import *


def convert_tensor(tensor):
    res = tensor[..., 0].unsqueeze(-1)
    return res
