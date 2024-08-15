import pickle

from NssMPC.common.network.tcp_process import TCPProcess
from NssMPC.common.utils import *
from NssMPC.config import SOCKET_NUM
from NssMPC.common.network.async_tcp import TCPServer, TCPClient


class Communicator:
    def __init__(self):
        self.address = None
        self.port = None
        self.comm_rounds = {'send': 0, 'recv': 0}
        self.comm_bytes = {'send': 0, 'recv': 0}
        self.max_connections = None
        self.server_socket = None
        self.sending_socket_mapping = {}

    def set_address(self, address, port):
        self.address = address
        self.port = port

    def set_max_connections(self, max_connections):
        self.max_connections = max_connections

    def init_server(self):
        raise NotImplementedError

    def connect_to_other(self, other_address, socket_address):
        raise NotImplementedError

    def connected_num(self):
        raise NotImplementedError

    def _send_to(self, address, x):
        raise NotImplementedError

    def _recv_from(self, address):
        raise NotImplementedError

    def _close(self):
        raise NotImplementedError

    def send_to_address(self, address, x):
        self._send_to(address, x)
        self.comm_rounds['send'] += 1
        self.comm_bytes['send'] += count_bytes(x)

    def recv_from_address(self, address):
        item = self._recv_from(address)
        self.comm_rounds['recv'] += 1
        self.comm_bytes['recv'] += count_bytes(item)
        return item

    def close(self):
        self._close()
        print(f"Communicator: {self.address} closed.")
        print(f"Communication costs:\n\tsend rounds: {self.comm_rounds['send']}\t\t"
              f"send bytes: {bytes_convert(self.comm_bytes['send'])}.")
        print(f"\trecv rounds: {self.comm_rounds['recv']}\t\t"
              f"recv bytes: {bytes_convert(self.comm_bytes['recv'])}.")


class MCommunicator(Communicator):
    """
    Multi-socket Communicator
    """

    def __init__(self):
        super(MCommunicator, self).__init__()
        self.tcp_process = None
        self.in_pipe = None
        self.out_pipe = None
        self.socket_num = SOCKET_NUM
        self.connected_parties = 0

    def set_socket_num(self, socket_num):
        self.socket_num = socket_num

    def init_server(self):
        self.tcp_process = TCPProcess(server_ip=(self.address, self.port))
        self.out_pipe, self.in_pipe = self.tcp_process.get_pipe()
        self.tcp_process.start((self.address, self.port))

    def connect_to_other(self, other_address, socket_address):
        self.out_pipe.send('CONNECT')
        self.out_pipe.send((other_address, socket_address))
        if self.out_pipe.recv() == 'OK':
            print("All sockets are connected to", other_address)
        else:
            raise ConnectionError("Failed to connect.")

    def connected_num(self):
        self.out_pipe.send('CHECK')
        return self.out_pipe.recv()

    def _send_to(self, address, x):
        self.out_pipe.send('SEND')
        self.out_pipe.send(address)
        msg = pickle.dumps(x)
        self.out_pipe.send(msg)

    def _recv_from(self, address):
        self.out_pipe.send('RECV')
        self.out_pipe.send(address)
        msg = self.out_pipe.recv()
        msg = pickle.loads(msg)
        return msg

    def _close(self):
        self.out_pipe.send('CLOSE')
        self.tcp_process.join()


class NCommunicator(Communicator):
    def __init__(self):
        super(NCommunicator, self).__init__()

    def connect_to_other(self, other_address, socket_address):
        client_socket = TCPClient(socket_address[0], socket_address[1])
        client_socket.connect_to_with_retry(other_address[0], other_address[1])
        self.sending_socket_mapping[str(other_address)] = client_socket

    def init_server(self):
        self.server_socket = TCPServer(self.address, self.port)
        self.server_socket.run()

    def connected_num(self):
        return self.server_socket.connect_number

    def _recv_from(self, address):
        item = self.server_socket.receive_serializable_item_from((str(address)))
        return item

    def _send_to(self, address, x):
        socket = self.sending_socket_mapping.get(str(address))
        socket.send_serializable_item(x)

    def _close(self):
        for address, socket in self.sending_socket_mapping.items():
            socket.close()
        self.server_socket.close_all()
