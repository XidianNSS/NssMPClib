import pickle

from common.network.tcp_process import TCPProcess
from common.utils import *


class Communicator:
    def __init__(self):
        self.tcp_process = None
        self.in_pipe = None
        self.out_pipe = None
        self.address = None
        self.server_port = None
        self.socket_num = None

        self.comm_rounds = {'send': 0, 'recv': 0}
        self.comm_bytes = {'send': 0, 'recv': 0}

    def set_address(self, address, server_port):
        self.address = address
        self.server_port = server_port

    def set_socket_num(self, socket_num):
        self.socket_num = socket_num

    def start_tcp_process(self, client_ip, target_server_ip, client_mapping):
        # TODO: assign port automatically
        self.tcp_process = TCPProcess(server_ip=(self.address, self.server_port), client_ip=client_ip,
                                      target_server_ip=target_server_ip, client_mapping=client_mapping)
        self.out_pipe, self.in_pipe = self.tcp_process.get_pipe()
        self.tcp_process.start()

    def connect(self):
        self.out_pipe.send('CONNECT')
        if self.out_pipe.recv() == 'OK':
            print("All parties are online.")
        else:
            raise ConnectionError("Failed to connect all parties.")

    def send_to(self, target, msg):
        self.comm_rounds['send'] += 1
        self.out_pipe.send('SEND')
        self.out_pipe.send(target)
        msg = pickle.dumps(msg)
        self.out_pipe.send(msg)
        self.comm_bytes['send'] += count_bytes(msg)

    def recv_from(self, source):
        self.comm_rounds['recv'] += 1
        self.out_pipe.send('RECV')
        self.out_pipe.send(source)
        msg = self.out_pipe.recv()
        msg = pickle.loads(msg)
        self.comm_bytes['recv'] += count_bytes(msg)
        return msg

    def close(self):
        self.out_pipe.send('CLOSE')
        self.tcp_process.join()
        print(f"Communicator: {self.address} closed.")
        print(f"Communication costs:\n\tsend rounds: {self.comm_rounds['send']}\t\t"
              f"send bytes: {bytes_convert(self.comm_bytes['send'])}.")
        print(f"\trecv rounds: {self.comm_rounds['recv']}\t\t"
              f"recv bytes: {bytes_convert(self.comm_bytes['recv'])}.")
