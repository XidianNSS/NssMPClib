import os.path

from NssMPC.config import param_path


class BaseParamProvider(object):
    def __init__(self, param_type, saved_name=None, param_tag=None, root_path=None):
        self.param_type = param_type
        self.param = None
        self.buffer = None
        self.in_pipe = None
        self.out_pipe = None
        self.lock = None

        self.ptr = 0
        self.buffer_ptr = 0
        self.infile_ptr = 0
        self.file_ptr = 0
        self.left_ptr = 0  # without buffer

        self.have_params = True
        self.num_of_party = 2
        if saved_name is None:
            self.saved_name = self.param_type.__name__
        else:
            self.saved_name = saved_name

        if param_tag is None:
            self.param_tag = self.param_type.__name__
        else:
            self.param_tag = param_tag
        if root_path is None:
            self.root_path = f"{param_path}{self.param_type.__name__}/"

    def set_pipe(self, pipe):
        self.out_pipe, self.in_pipe = pipe

    def set_in_pipe(self, in_pipe):
        self.in_pipe = in_pipe

    def set_out_pipe(self, out_pipe):
        self.out_pipe = out_pipe

    def set_lock(self, lock):
        self.lock = lock

    def load_param(self, saved_name=None):
        raise NotImplementedError

    def load_buffers(self, party_id, saved_name=None):
        if saved_name is None:
            file_name = f'{self.saved_name}_{self.file_ptr}_{party_id}.pth'
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

    def get_parameters(self, number_of_params):
        raise NotImplementedError

    def update_params(self, number_of_params):
        # Shift existing triples
        shift_size = self.left_ptr - self.ptr

        # Check for sufficient buffer
        self._ensure_buffer_availability(number_of_params, shift_size)
        with self.lock:
            self.param[:shift_size] = self.param[self.ptr:self.left_ptr].clone()
            # Update from buffer
            end = min(shift_size + self.buffer_ptr, self.left_ptr)
            self._update_from_buffer(shift_size, end)

            # Reset pointers
            self.ptr = 0
            self.left_ptr = end

            # Notify for more triples
            self.in_pipe.send(1)

    def _ensure_buffer_availability(self, number_of_params, shift_size):
        if number_of_params > shift_size + self.buffer_ptr:
            self.in_pipe.send('need params')
            if self.in_pipe.recv() == 0:
                self.in_pipe.close()
                raise Exception(f"Have no more parameters {self.param.__class__.__name__}! Need to generate!")

    def _update_from_buffer(self, start, end):
        buffer_left = end - start
        self.param[start:end] = self.buffer[:buffer_left]

        # Update and resize buffer
        self._rebuild_buffer(buffer_left)

    def _rebuild_buffer(self, buffer_left):
        temp = self.buffer

        self.buffer = self.param.clone()

        self.buffer[:self.buffer_ptr - buffer_left] = temp[buffer_left:self.buffer_ptr]

        self.buffer_ptr = self.buffer_ptr - buffer_left
