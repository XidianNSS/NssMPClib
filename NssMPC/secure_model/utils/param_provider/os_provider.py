#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import os.path

from NssMPC.secure_model.utils.param_provider.param_provider import ParamProvider


class OSProvider(ParamProvider):
    def __init__(self, param_type, saved_name=None, param_tag=None, root_path=None):
        """
        Initialize the OSProvider class.

        :param param_type: Type of parameter
        :type param_type: Type
        :param saved_name: Optional file name to save.
        :type saved_name: str
        :param param_tag: A label used to identify parameters
        :type param_tag: str
        :param root_path: The root path of the parameter file
        :type root_path: str
        """
        super().__init__(param_type, saved_name, param_tag, root_path)

    def load_buffers(self, party_id, saved_name=None):
        """
        Load the parameters in the buffer.

        First build the file name:
            If no saved_name is provided, the file name in the format of ``{saved_name}_{file_ptr}_{party_id}.pth`` is generated.

        In the case that the file exists and there are still loadable arguments, the data is loaded in a loop:
            First load the arguments from the specified path to ``buffer_params``, then copy the loaded arguments into ``self.buffer``, and update the pointer. If ``infile_ptr`` reaches the length of ``buffer_params``, reset the pointer and update ``file_ptr`` to load the next file. If the file does not exist, set ``self.have_params`` to **False** to indicate that no more parameters are available.

        :param party_id: Identify the participant currently loading the parameters.
        :type party_id: int
        :param saved_name: Optional file name to save.
        :type saved_name: str
        """
        if saved_name is None:
            file_rsplit = self.saved_name.rsplit("_", 1)
            file_name = file_rsplit[0] + "_" + str(self.file_ptr) + "_" + file_rsplit[-1] + ".pth"
        else:
            file_name = saved_name
        param_read_path = f"{self.root_path}"
        num_of_params = len(self.param) - self.buffer_ptr
        with self.lock:
            while num_of_params > 0 and os.path.exists(param_read_path + file_name):
                buffer_params = self.param_type.load(param_read_path + file_name)

                copy_num = min(num_of_params, len(buffer_params) - self.infile_ptr)
                end = self.buffer_ptr + copy_num
                self.buffer[self.buffer_ptr: end] = buffer_params[self.infile_ptr: self.infile_ptr + copy_num]
                self.buffer_ptr = end
                num_of_params -= copy_num
                self.infile_ptr += copy_num

                if self.infile_ptr == len(buffer_params):
                    self.infile_ptr = 0
                    self.file_ptr += 1
                    file_rsplit = file_name.rsplit("_", 2)
                    file_name = file_rsplit[0] + "_" + str(self.file_ptr) + "_" + file_rsplit[-1] + ".pth"

            if not os.path.exists(param_read_path + file_name):
                self.have_params = False
                return
