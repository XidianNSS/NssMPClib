#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch

from NssMPC import RingTensor
from NssMPC.config import data_type
from NssMPC.crypto.aux_parameter import Parameter


class Wrap(Parameter):
    """
    A class used for operations related to the truncation, including methods for generating relevant parameters and calculating the number of wraps.
    """

    def __init__(self, r=None, theta_r=None):
        """
        Attribute:
            *  **r** (*ArithmeticSecretSharing*): Generated random numbers, used to obfuscate real input data in the secure truncation protocol.
            *  **theta_r** (*ArithmeticSecretSharing*): Parameters used in truncation
        """
        self.r = r
        self.theta_r = theta_r

    @staticmethod
    def gen(num_of_params):
        """
        Generates parameters for wrap-related operations.

        :param num_of_params: The number of params to generate.
        :type num_of_params: int
        :return: The generated parameters for two parties.
        :rtype: Wrap
        """
        from NssMPC.crypto.primitives.arithmetic_secret_sharing import ArithmeticSecretSharing
        r = RingTensor.random([num_of_params])
        r0, r1 = ArithmeticSecretSharing.share(r, 2)
        theta_r = Wrap.count_wraps([r0.item.tensor, r1.item.tensor])

        theta_r0, theta_r1 = ArithmeticSecretSharing.share(RingTensor(theta_r), 2)

        wrap_0 = Wrap(r0.item.tensor, theta_r0.item.tensor)
        wrap_1 = Wrap(r1.item.tensor, theta_r1.item.tensor)

        return wrap_0, wrap_1

    @staticmethod
    def count_wraps(share_list):
        """
        Computes the number of overflows or underflows in a set of shares

        We compute this by counting the number of overflows and underflows as we
        traverse the list of shares.

        :param share_list: The list contains the shares to compute warps.
        :type share_list: list
        :return: The number of overflows or underflows, with overflows being positive and underflows being negative.
        :rtype: torch.Tensor
        """
        result = torch.zeros_like(share_list[0], dtype=data_type)
        prev = share_list[0]
        for cur in share_list[1:]:
            next = cur + prev
            result -= ((prev < 0) & (cur < 0) & (next > 0)).to(data_type)  # underflow
            result += ((prev > 0) & (cur > 0) & (next < 0)).to(data_type)  # overflow
            prev = next
        return result
