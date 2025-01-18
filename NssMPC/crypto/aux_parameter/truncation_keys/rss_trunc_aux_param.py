#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import os
from NssMPC.common.ring import RingTensor
from NssMPC.crypto.aux_parameter import Parameter
from NssMPC.config.configs import SCALE, param_path


class RssTruncAuxParams(Parameter):
    """
    A class used for generating relevant parameters for truncation of RSS and save the parameters.
    """

    def __init__(self):
        """
        ATTRIBUTES:
            * **r** (:class:`~NssMPC.common.ring.ring_tensor.RingTensor`): The auxiliary parameter for truncation of RSS.
            * **r_hat** (:class:`~NssMPC.common.ring.ring_tensor.RingTensor`): The auxiliary parameter for truncation of RSS.
            * **size** (*int*): Size of the parameters.
        """
        self.r = None
        self.r_hat = None
        self.size = 0

    def __iter__(self):
        """
        Make an instance of this class iterable.

        :return: A tuple contains the attributes r and r_hat.
        :rtype: tuple
        """
        return iter((self.r, self.r_hat))

    @staticmethod
    def gen(num_of_params, scale=SCALE):
        """
        Generates parameters for truncation operations of RSS.

        :param num_of_params: The number of params to generate.
        :type num_of_params: int
        :param scale: The scale of the number to be truncated, defaults to SCALE.
        :type scale: int
        :return: The generated parameters for three parties.
        :rtype: List[RssTruncAuxParams]
        """
        from NssMPC.crypto.primitives.arithmetic_secret_sharing import ReplicatedSecretSharing
        from NssMPC.config.configs import HALF_RING
        r_hat = RingTensor.random([num_of_params], down_bound=-HALF_RING // (2 * scale),
                                  upper_bound=HALF_RING // (2 * scale))
        r = r_hat * scale
        r_list = ReplicatedSecretSharing.share(r)
        r_hat_list = ReplicatedSecretSharing.share(r_hat)
        aux_params = []
        for i in range(3):
            param = RssTruncAuxParams()
            param.r = r_list[i].to('cpu')
            param.r_hat = r_hat_list[i].to('cpu')
            param.size = num_of_params
            aux_params.append(param)
        return aux_params

    @classmethod
    def gen_and_save(cls, num, scale=SCALE):
        """
        Generates parameters for truncation operations of RSS and save them.

        :param num: The number of params to generate.
        :type num: int
        :param scale: The scale of the number to be truncated, defaults to SCALE.
        :type scale: int
        :return: The generated parameters for three parties.
        :rtype: List[RssTruncAuxParams]
        """
        aux_params = cls.gen(num, scale)
        file_path = f"{param_path}{cls.__name__}/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        file_name_base = f"RssTruncAuxParams_" if scale == SCALE else f"RssTruncAuxParams_{scale}_"
        for i in range(3):
            file_name = f"{file_name_base}{i}.pth"
            aux_params[i].save(file_path, file_name)
