from multiprocessing import Pipe
from threading import Thread, Lock

from common.aux_parameter.param_provider import ParamProvider


class BufferThread(object):
    def __init__(self, params_provider: ParamProvider, party, pipe=Pipe(True), lock=Lock()):
        self.triples_provider = params_provider
        self.party = party
        self.out_pipe, self.in_pipe = pipe
        self.triples_provider.set_pipe((self.out_pipe, self.in_pipe))
        self.triples_provider.set_lock(lock)
        self.sub_p = Thread(target=provider_thread,
                            args=(self.triples_provider, (self.out_pipe, self.in_pipe)))

    def start(self):
        self.sub_p.start()

    def get_pipe(self):
        return self.out_pipe, self.in_pipe

    def join(self):
        self.in_pipe.close()
        self.out_pipe.close()
        self.sub_p.join()


def provider_thread(params_buffer, pipe):
    _out_pipe, _in_pipe = pipe

    while True:
        try:
            params_buffer.load_buffers()
            msg = _out_pipe.recv()
            if msg == 'need triples':
                if params_buffer.have_params:
                    _out_pipe.send(1)
                else:
                    _out_pipe.send(0)
                    _out_pipe.close()
                    _in_pipe.close()
                    break
        except EOFError:
            break
