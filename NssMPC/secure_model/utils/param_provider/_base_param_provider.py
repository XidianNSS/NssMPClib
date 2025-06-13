#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
Used to load and store basic parameters.
"""

import os.path

from NssMPC.config import param_path


class BaseParamProvider(object):
    """
    A flexible and extensible base class for handling parameter loading and management.
    """

    def __init__(self, param_type, saved_name=None, param_tag=None, root_path=None):
        """
        ATTRIBUTES:
            * **param_type** (*Type*): A class or object that describes the type of parameters.
            * **param** (*Any*): Used to store parameters.
            * **buffer** (*list*): A buffer used to store parameters.
            * **in_pipe** (*Pipe*): Used for inputting data
            * **out_pipe** (*Pipe*): Used for outputting data.
            * **lock** (*Lock*): Ensure thread safety.
            * **ptr** (*int*): Points to the current parameter position.
            * **buffer_ptr** (*int*): Point to the location of the pointed buffer.
            * **infile_ptr** (*int*): Point to the location of the input file.
            * **file_ptr** (*int*): Point to the current file location.
            * **left_ptr** (*itn*): Point to the remaining parameter positions.
            * **have_params** (*bool*): Whether there are still available parameters
            * **num_of_party** (*int*): Number of parties, (default is **2**)
            * **root_path** (*str*): The root path of the parameter file.

        """
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
        """
        Set up input and output pipelines.

        :param pipe: Communication pipe
        :type pipe: tuple
        """
        self.out_pipe, self.in_pipe = pipe

    def set_in_pipe(self, in_pipe):
        """
        Set up input pipelines.

        :param in_pipe: Communication in pipe
        :type in_pipe: Pipe
        """
        self.in_pipe = in_pipe

    def set_out_pipe(self, out_pipe):
        """
        Set up output pipelines.

        :param out_pipe: Communication out pipe
        :type out_pipe: Pipe
        """
        self.out_pipe = out_pipe

    def set_lock(self, lock):
        """
        Set lock.

        :param lock: Used for thread or process synchronization to ensure safe access to shared resources.
        :type lock: Lock
        """
        self.lock = lock

    def load_param(self, saved_name=None):
        raise NotImplementedError

    def load_buffers(self, party_id, saved_name=None):
        """
        Load the parameters in the buffer.

        First build the file name:
            If no saved_name is provided, the file name in the format of ``{saved_name}_{file_ptr}_{party_id}.pth`` is generated.

        In the case that the file exists and there are still loadable arguments, the data is loaded in a loop:
            first load the arguments from the specified path to ``buffer_params``, then copy the loaded arguments into ``self.buffer``, and update the pointer. If ``infile_ptr`` reaches the length of ``buffer_params``, reset the pointer and update ``file_ptr`` to load the next file. If the file does not exist, set ``self.have_params`` to **False** to indicate that no more parameters are available.

        :param party_id: Identify the participant currently loading the parameters.
        :type party_id: int
        :param saved_name: Optional file name to save.
        :type saved_name: str
        """
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
        """
        Get parameters.

        :param number_of_params: the number of parameters.
        :type number_of_params: int
        """
        raise NotImplementedError

    def update_params(self, number_of_params):
        """
        Update the buffer for storing storage parameters.

        First, copy the portion of existing data from ``self.ptr`` to ``self.left_ptr`` to the start location of ``self.param``.

        .. note::
            * The purpose of this is to move existing data to the beginning of the buffer, clearing up space for new data.
            * ``left_ptr`` Indicates the end position of the actual data in the buffer

        Then use :meth:`_update_from_buffer` to update the parameters in the buffer.

        :param number_of_params: the number of parameters.
        :type number_of_params: int
        """
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
        """
        Ensure that there are enough parameters available in the buffer.

        First check the availability of the buffer: whether the number of parameters required is greater than the sum
        of the current buffer pointer and shift sizes. If the parameters are insufficient, a message is sent
        requesting that the parameters wait to receive data from the input pipeline. If a value of 0 is received,
        no more parameters are available. In this case, close the pipe and throw an exception.

        :param number_of_params: The number of required parameters.
        :type number_of_params: int
        :param shift_size: The current size of the available buffer.
        :type shift_size: int
        :return:
        """
        if number_of_params > shift_size + self.buffer_ptr:
            self.in_pipe.send('need params')
            if self.in_pipe.recv() == 0:
                self.in_pipe.close()
                raise Exception(f"Have no more parameters {self.param.__class__.__name__}! Need to generate!")

    def _update_from_buffer(self, start, end):
        """
        Update parameters from the buffer.

        After getting the number of parameters to update, copy the data in the buffer to the specified location in
        ``self.param``, and then adjust the buffer using the method :meth:`_rebuild_buffer`

        :param start: The starting point for the update
        :type start: int
        :param end: The ending point for the update
        :type end: int
        """
        buffer_left = end - start
        self.param[start:end] = self.buffer[:buffer_left]

        # Update and resize buffer
        self._rebuild_buffer(buffer_left)

    def _rebuild_buffer(self, buffer_left):
        """
        Rebuild the buffer to accommodate the new parameter state.

        After saving the current buffer to temp, copy ``self.param`` to the buffer ``self.buffer`` and copy the old buffer
        data to the appropriate location in the new buffer so that unused parameters are preserved. Finally, update
        the ``buffer_ptr`` pointer.

        :param buffer_left: The number of parameters transferred from the buffer
        :type buffer_left: int
        """
        temp = self.buffer

        self.buffer = self.param.clone()

        self.buffer[:self.buffer_ptr - buffer_left] = temp[buffer_left:self.buffer_ptr]

        self.buffer_ptr = self.buffer_ptr - buffer_left
