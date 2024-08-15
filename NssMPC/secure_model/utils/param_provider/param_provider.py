import os.path

from NssMPC.config import DEVICE, DEBUG_LEVEL
from NssMPC.secure_model.utils.param_provider._base_param_provider import BaseParamProvider


class ParamProvider(BaseParamProvider):
    def __init__(self, param_type, saved_name=None, param_tag=None, root_path=None):
        super().__init__(param_type, saved_name, param_tag, root_path)

    def load_param(self, saved_name=None):
        if saved_name is None:
            file_name = f"{self.saved_name}.pth"
        else:
            file_name = saved_name
        file_path = os.path.join(self.root_path, file_name)
        self.param = self.param_type.load(file_path=file_path)
        self.buffer = self.param.clone()
        self.left_ptr = len(self.param)

    def get_parameters(self, number_of_params):
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
