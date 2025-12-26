#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
Load, manage, and acquire the parameters needed for computation.
"""
import os.path

from nssmpc.config import DEVICE, DEBUG_LEVEL, param_path


class BaseParamProvider(object):
    """
    A flexible and extensible base class for handling parameter loading and management.

    Attributes:
        param_type (Type): A class or object that describes the type of parameters.
        param (Any): Used to store parameters.
        buffer (list): A buffer used to store parameters.
        in_pipe (Pipe): Used for inputting data.
        out_pipe (Pipe): Used for outputting data.
        lock (Lock): Ensure thread safety.
        ptr (int): Points to the current parameter position.
        buffer_ptr (int): Point to the location of the pointed buffer.
        infile_ptr (int): Point to the location of the input file.
        file_ptr (int): Point to the current file location.
        left_ptr (int): Point to the remaining parameter positions.
        have_params (bool): Whether there are still available parameters.
        num_of_party (int): Number of parties, (default is **2**).
        root_path (str): The root path of the parameter file.
    """

    def __init__(self, param_type, saved_name=None, root_path=None):
        """
        Initialize the BaseParamProvider.

        Args:
            param_type (Type): A class or object that describes the type of parameters.
            saved_name (str, optional): The name used for saving parameters. Defaults to None.
            root_path (str, optional): The root directory for parameter files. Defaults to None.

        Examples:
            >>> provider = BaseParamProvider(MyParamType)
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

        if root_path is None:
            self.root_path = f"{param_path}{self.param_type.__name__}/"

    def set_pipe(self, pipe):
        """
        Set up input and output pipelines.

        Args:
            pipe (tuple): Communication pipe containing (out_pipe, in_pipe).

        Examples:
            >>> provider.set_pipe((out_pipe, in_pipe))
        """
        self.out_pipe, self.in_pipe = pipe

    def set_in_pipe(self, in_pipe):
        """
        Set up input pipelines.

        Args:
            in_pipe (Pipe): Communication in pipe.

        Examples:
            >>> provider.set_in_pipe(in_pipe)
        """
        self.in_pipe = in_pipe

    def set_out_pipe(self, out_pipe):
        """
        Set up output pipelines.

        Args:
            out_pipe (Pipe): Communication out pipe.

        Examples:
            >>> provider.set_out_pipe(out_pipe)
        """
        self.out_pipe = out_pipe

    def set_lock(self, lock):
        """
        Set lock.

        Args:
            lock (Lock): Used for thread or process synchronization to ensure safe access to shared resources.

        Examples:
            >>> provider.set_lock(lock)
        """
        self.lock = lock

    def load_param_from_file(self, *args):
        """
        Load parameters.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def load_buffers(self, party_id, saved_name=None):
        """
        Load the parameters in the buffer.

        First build the file name:
            If no saved_name is provided, the file name in the format of ``p{party_id}_{saved_name}_{file_ptr}.pth`` is generated.

        In the case that the file exists and there are still loadable arguments, the data is loaded in a loop:
            first load the arguments from the specified path to ``buffer_params``, then copy the loaded arguments into ``self.buffer``, and update the pointer. If ``infile_ptr`` reaches the length of ``buffer_params``, reset the pointer and update ``file_ptr`` to load the next file. If the file does not exist, set ``self.have_params`` to **False** to indicate that no more parameters are available.

        Args:
            party_id (int): Identify the participant currently loading the parameters.
            saved_name (str, optional): Optional file name to save. Defaults to None.

        Examples:
            >>> provider.load_buffers(0)
        """
        if saved_name is None:
            file_name = f'p{party_id}_{self.saved_name}_{self.file_ptr}.pth'
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

    def get_parameters(self, *args):
        """
        Get parameters.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def update_params(self, number_of_params):
        """
        Update the buffer for storing storage parameters.

        First, copy the portion of existing data from ``self.ptr`` to ``self.left_ptr`` to the start location of ``self.param``.

        Note:
            * The purpose of this is to move existing data to the beginning of the buffer, clearing up space for new data.
            * ``left_ptr`` Indicates the end position of the actual data in the buffer

        Then use ``_update_from_buffer`` to update the parameters in the buffer.

        Args:
            number_of_params (int): The number of parameters.

        Examples:
            >>> provider.update_params(100)
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

        Args:
            number_of_params (int): The number of required parameters.
            shift_size (int): The current size of the available buffer.

        Raises:
            Exception: If no more parameters are available.

        Examples:
            >>> provider._ensure_buffer_availability(100, 50)
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
        ``self.param``, and then adjust the buffer using the method ``_rebuild_buffer``.

        Args:
            start (int): The starting point for the update.
            end (int): The ending point for the update.

        Examples:
            >>> provider._update_from_buffer(0, 100)
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

        Args:
            buffer_left (int): The number of parameters transferred from the buffer.

        Examples:
            >>> provider._rebuild_buffer(50)
        """
        temp = self.buffer

        self.buffer = self.param.clone()

        self.buffer[:self.buffer_ptr - buffer_left] = temp[buffer_left:self.buffer_ptr]

        self.buffer_ptr = self.buffer_ptr - buffer_left


class ParamProvider(BaseParamProvider):
    """
    Gets parameters from the buffer.
    """

    def __init__(self, param_type, saved_name=None, root_path=None):
        """
        Initialize the ParamProvider.

        Args:
            param_type (Parameter): The type of parameters determines how they are loaded and processed.
            saved_name (str, optional): The file name used to save the parameters. Defaults to None.
            root_path (str, optional): The root path to store the parameter file. Defaults to None.

        Examples:
            >>> provider = ParamProvider(MyParamType)
        """
        super().__init__(param_type, saved_name, root_path)

    def load_param_from_file(self, tag=None):
        """
        Load parameters from file.

        If no tag is provided, the file name based on the tag attribute is used by default with the .pth
        suffix. Build the full file path and load the parameters, creating a copy of ``self.param`` into ``self.buffer``,
        ``self.left_ptr`` is set to the length of the argument and is used to track the number of remaining available
        arguments.

        Args:
            tag (str, optional): The file name used to save the parameters. Defaults to None.

        Examples:
            >>> provider.load_param_from_file()
        """
        if tag is None:
            file_name = f"{self.saved_name}.pth"
        else:
            file_name = tag
        file_path = os.path.join(self.root_path, file_name)
        self.param = self.param_type.load(file_path=file_path)
        if DEBUG_LEVEL == 2:
            self.param = self.param.to('cuda')
        self.buffer = self.param.clone()
        self.left_ptr = len(self.param)

    def get_parameters(self, number_of_params):
        """
        Get parameters.

        First check if the parameters have been loaded, and throw an exception if they have not.
        If DEBUG_LEVEL is 2, return the first parameter directly.
        Check whether the number of parameters requested exceeds the length of the available parameters, and if so, throw an exception.
        If the current pointer plus the number of requested parameters exceeds the number of remaining available parameters, the update_params method is called to update the parameters.
        Extract the desired number of parameters from ``self.param`` and update the pointer.

        Args:
            number_of_params (int): The number of parameters.

        Returns:
            list: Extracted parameters.

        Raises:
            Exception: If parameters are not loaded or buffer size is insufficient.

        Examples:
            >>> params = provider.get_parameters(10)
        """
        if self.param is None:
            raise Exception(f"{self.__class__.__name__}:Please load parameters first!")
        if DEBUG_LEVEL == 2:
            return self.param[0].to(DEVICE)
        if number_of_params > len(self.param):
            raise Exception(
                f"{self.__class__.__name__}:The number of parameters is larger than the size of the buffer! need {number_of_params}, but only {len(self.param)} left!")
        if self.ptr + number_of_params > self.left_ptr:
            self.update_params(number_of_params)

        param = self.param[self.ptr:self.ptr + number_of_params]
        if number_of_params == 1:
            param = param[0]

        if not DEBUG_LEVEL:
            self.ptr += number_of_params
        return param.to(DEVICE)
