#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
Load, manage, and acquire the parameters needed for computation.
"""
import os.path

from NssMPC.config import DEVICE, DEBUG_LEVEL
from NssMPC.secure_model.utils.param_provider._base_param_provider import BaseParamProvider


class ParamProvider(BaseParamProvider):
    """
    Gets parameters from the buffer.
    """

    def __init__(self, param_type, saved_name=None, param_tag=None, root_path=None):
        """
        :param param_type: The type of parameters determines how they are loaded and processed.
        :type param_type: str or type
        :param saved_name: The file name used to save the parameters.
        :type saved_name: str
        :param param_tag: A label used to identify parameters
        :type param_tag: str
        :param root_path: The root path to store the parameter file
        :type root_path: str
        """
        super().__init__(param_type, saved_name, param_tag, root_path)

    def load_param(self, saved_name=None):
        """
        If no saved_name is provided, the file name based on the saved_name attribute is used by default with the.pth
        suffix. Build the full file path and load the parameters, creating a copy of ``self.param`` into ``self.buffer``,
        ``self.left_ptr`` is set to the length of the argument and is used to track the number of remaining available
        arguments.

        :param saved_name: The file name used to save the parameters.
        :type saved_name: str
        """
        if saved_name is None:
            file_name = f"{self.saved_name}.pth"
        else:
            file_name = saved_name
        file_path = os.path.join(self.root_path, file_name)
        self.param = self.param_type.load(file_path=file_path)
        self.buffer = self.param.clone()
        self.left_ptr = len(self.param)

    def get_parameters(self, number_of_params):
        """
        Get parameter.

        First check if the parameters have been loaded, and throw an exception if they have not.

        If DEBUG_LEVEL is 2, return the first parameter directly.

        Check whether the number of parameters requested exceeds the length of the available parameters, and if so, throw an exception.

        If the current pointer plus the number of requested parameters exceeds the number of remaining available parameters, the update_params method is called to update the parameters.

        Extract the desired number of parameters from ``self.param`` and update the pointer.

        :param number_of_params: the number of parameters
        :type number_of_params: int
        :return: Extracted parameters
        :rtype: list
        """
        if self.param is None:
            raise Exception(f"{self.__class__.__name__}:Please load parameters first!")
        if DEBUG_LEVEL == 2:
            return self.param[0].to(DEVICE)
        if number_of_params > len(self.param):
            raise Exception(
                f"{self.__class__.__name__}:The number of parameters is larger than the size of the buffer!")
        if self.ptr + number_of_params > self.left_ptr:
            self.update_params(number_of_params)

        param = self.param[self.ptr:self.ptr + number_of_params]

        if not DEBUG_LEVEL:
            self.ptr += number_of_params
        # todo:这里似乎无法指定party
        return param.to(DEVICE)
