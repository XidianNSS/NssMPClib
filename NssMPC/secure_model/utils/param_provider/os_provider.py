import os.path

from NssMPC.secure_model.utils.param_provider.param_provider import ParamProvider


class OSProvider(ParamProvider):
    def __init__(self, param_type, saved_name=None, param_tag=None, root_path=None):
        super().__init__(param_type, saved_name, param_tag, root_path)

    def load_buffers(self, party_id, saved_name=None):
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
