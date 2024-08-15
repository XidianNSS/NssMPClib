from multiprocessing import Pipe
from threading import Thread, Lock

from NssMPC.secure_model.utils.param_provider.param_provider import BaseParamProvider


class BufferThread(object):
    def __init__(self, params_provider: BaseParamProvider, party, pipe=Pipe(True), lock=Lock()):
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
        self.sub_p.start()

    def get_pipe(self):
        return self.out_pipe, self.in_pipe

    def join(self):
        self.in_pipe.close()
        self.out_pipe.close()
        self.sub_p.join()

    def set_buffer_file_name(self, file_name):
        self.buffer_file_name = file_name


def provider_thread(params_buffer, pipe, party_id, buffer_file_name):
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
