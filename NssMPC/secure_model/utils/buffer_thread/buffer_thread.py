#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from multiprocessing import Pipe
from threading import Thread, Lock

from NssMPC.secure_model.utils.param_provider import BaseParamProvider


class BufferThread(object):
    """
    Load and manage parameters in the buffer
    """

    def __init__(self, params_provider: BaseParamProvider, party, pipe=Pipe(True), lock=Lock()):
        """
        ATTRIBUTES:
            * **params_provider** (:class:`~NssMPC.secure_model.utils.param_provider._base_param_provider.BaseParamProvider`): An object that provides parameters.
            * **party** (:class:`~NssMPC.secure_model.mpc_party.party.Party`): Indicate the participant to which the current thread belongs.
            * **in_pipe** (*Pipe*): Used for inputting data
            * **out_pipe** (*Pipe*): Used for outputting data.
            * **buffer_file_name** (*str*): The name of the buffer file.
            * **sub_p** (*Thread*): A new thread passes the necessary parameters.

        """
        self.params_provider = params_provider
        self.party = party
        self.out_pipe, self.in_pipe = pipe
        self.params_provider.set_pipe((self.out_pipe, self.in_pipe))
        self.params_provider.set_lock(lock)
        self.buffer_file_name = None
        self.sub_p = Thread(target=provider_thread,
                            args=(self.params_provider, (self.out_pipe, self.in_pipe), self.party.party_id,
                                  self.buffer_file_name))

    def start(self):
        """
        Start the subthread ``sub_p``.
        """
        self.sub_p.start()

    def get_pipe(self):
        """
        :return: the output and input parts of the pipe, allowing for external access.
        """
        return self.out_pipe, self.in_pipe

    def join(self):
        """
        Close the input and output pipes to ensure that no further message passing occurs.

        Wait for the completion of the execution of ``sub_p`` thread.
        """
        self.in_pipe.close()
        self.out_pipe.close()
        self.sub_p.join()

    def set_buffer_file_name(self, file_name):
        """
        Sets the buffer file name to be loaded, providing flexibility to specify different parameter files.

        :param file_name: File name
        :type file_name: str
        """
        self.buffer_file_name = file_name


def provider_thread(params_buffer, pipe, party_id, buffer_file_name):
    """
    First call the :meth:`~NssMPC.secure_model.utils.param_provider._base_param_provider.BaseParamProvider.load_buffers` method to load the initial parameters, then loop around waiting for a message from ``_out_pipe`` if you receive a message that requires parameters:

        If any parameters are available, call :meth:`~NssMPC.secure_model.utils.param_provider._base_param_provider.BaseParamProvider.load_buffers` to load the parameters and send a confirmation signal (1)

        If no parameters are available, send a signal (0), then close the output pipeline and exit the loop.

    Catch the **EOFError** exception and exit the loop when the pipe is closed.

    :param params_buffer: The parameters passed in are used to create an instance of the class.
    :type params_buffer: BaseParamProvider
    :param pipe: A pipe used to communicate with the main thread
    :type pipe: Pipe
    :param party_id: Identify the participant currently loading the parameters.
    :type party_id: int
    :param buffer_file_name: The name of the buffer file.
    :type buffer_file_name: str
    """
    _out_pipe, _in_pipe = pipe
    params_buffer.load_buffers(party_id, buffer_file_name)
    while True:
        try:
            msg = _out_pipe.recv()
            if msg == 'need params':
                if params_buffer.have_params:
                    params_buffer.load_buffers(party_id, buffer_file_name)
                    _out_pipe.send(1)
                else:
                    _out_pipe.send(0)
                    _out_pipe.close()
                    break
            else:
                params_buffer.load_buffers(party_id, buffer_file_name)
        except EOFError:
            break
