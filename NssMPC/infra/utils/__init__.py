#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.


def convert_tensor(tensor):
    res = tensor[..., 0].unsqueeze(-1)
    return res
