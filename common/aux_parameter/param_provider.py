import os.path

from config.base_configs import DEVICE, base_path, DEBUG_LEVEL


class ParamProvider(object):
    def __init__(self, param_type=None, party=None):
        self.param_type = param_type
        self.param = None
        self.load_buffer = None
        self.in_pipe = None
        self.out_pipe = None
        self.lock = None

        self.ptr = 0
        self.buffer_ptr = 0
        self.infile_ptr = 0
        self.file_ptr = 0
        self.left_ptr = 0  # without buffer

        self.have_params = True
        self.party = party
        self.num_of_party = 2

    def set_pipe(self, pipe):
        self.out_pipe, self.in_pipe = pipe

    def set_in_pipe(self, in_pipe):
        self.in_pipe = in_pipe

    def set_out_pipe(self, out_pipe):
        self.out_pipe = out_pipe

    def set_lock(self, lock):
        self.lock = lock

    def set_party(self, party):
        self.party = party

    def load_param(self):
        self.param = self.param_type.load(self.party.party_id, self.num_of_party)
        if hasattr(self.param, 'set_party'):
            self.param.set_party(self.party)
        self.load_buffer = self.param.clone()
        self.left_ptr = len(self.param)

    def load_buffers(self):
        param_path = f"{base_path}/aux_parameters/{self.param_type.__name__}/{self.num_of_party}party/"
        file_name = f'{self.param_type.__name__}_{self.party.party_id}_{self.file_ptr}.pth'

        num_of_params = len(self.param) - self.buffer_ptr

        with self.lock:
            while num_of_params > 0 and os.path.exists(param_path + file_name):
                buffer_params = self.param_type.load_by_name(file_name, param_path)

                copy_num = min(num_of_params, len(buffer_params) - self.infile_ptr)
                end = self.buffer_ptr + copy_num

                self.load_buffer[self.buffer_ptr: end] = buffer_params[self.infile_ptr: self.infile_ptr + copy_num]

                self.buffer_ptr = end
                num_of_params -= copy_num
                self.infile_ptr += copy_num

                if self.infile_ptr == len(buffer_params):
                    self.infile_ptr = 0
                    self.file_ptr += 1
                    file_name = file_name.rsplit("_", 1)[0] + "_" + str(self.file_ptr) + ".pth"

            if not os.path.exists(param_path + file_name):
                self.have_params = False
                return

    def get_parameters(self, number_of_params):
        if self.param is None:
            raise Exception(f"{self.__class__.__name__}:Please load parameters first!")
        if DEBUG_LEVEL == 2:
            param = self.param[0].to(DEVICE)
            return param
        if number_of_params > len(self.param):
            raise Exception(
                f"{self.__class__.__name__}:The number of parameters is larger than the size of the buffer!")
        if self.ptr + number_of_params > self.left_ptr:
            self.update_params(number_of_params)

        param = self.param[self.ptr:self.ptr + number_of_params]

        if not DEBUG_LEVEL:
            self.ptr += number_of_params

        return param.to(DEVICE)

    def update_params(self, number_of_params):
        with self.lock:
            # Shift existing triples
            shift_size = self.left_ptr - self.ptr
            self.param[:shift_size] = self.param[self.ptr:self.left_ptr]

            # Check for sufficient buffer
            self._ensure_buffer_availability(number_of_params, shift_size)

            # Update from buffer
            end = min(shift_size + self.buffer_ptr, self.left_ptr)
            self._update_from_buffer(shift_size, end)

            # Reset pointers
            self.ptr = 0
            self.left_ptr = end

            # Notify for more triples
            self.in_pipe.send(1)

    def _ensure_buffer_availability(self, number_of_triples, shift_size):
        if number_of_triples > shift_size + self.buffer_ptr:
            self.in_pipe.send('need triples')
            if self.in_pipe.recv() == 0:
                raise Exception(f"{self.__class__.__name__} have no parameters! Need to generate more!")

    def _update_from_buffer(self, start, end):
        buffer_left = end - start
        self.param[start:end] = self.load_buffer[:buffer_left]

        # Update and resize buffer
        self._rebuild_buffer(buffer_left)

    def _rebuild_buffer(self, buffer_left):
        temp = self.load_buffer

        self.load_buffer = self.param.clone()

        self.load_buffer[:self.buffer_ptr - buffer_left] = temp[buffer_left:self.buffer_ptr]

        self.buffer_ptr = self.buffer_ptr - buffer_left
