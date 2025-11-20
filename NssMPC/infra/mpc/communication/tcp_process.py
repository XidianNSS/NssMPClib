"""
Communication process
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from multiprocessing import Pipe
from threading import Thread, Lock

from NssMPC.infra.mpc.communication.multi_sockets_tcp import *


class TCPProcess(object):
    """Manages the communication process operations.

    This class provides the related operations involved in the communication process.
    """

    def __init__(self, server_ip, pipe=Pipe(True), lock=Lock(), socket_num=SOCKET_NUM):
        """Initializes a TCPProcess object.

        Args:
            server_ip (tuple): IP address of the server.
            pipe (tuple): Two-way communication pipeline.
            lock (Lock): Synchronization lock for shared resources.
            socket_num (int): The number of sockets.

        Examples:
            >>> process = TCPProcess(('127.0.0.1', 8000))
        """
        self.out_pipe, self.in_pipe = pipe
        self.lock = lock
        self.socket_num = socket_num
        self.sub_p = None
        self.server_ip = server_ip

    def start(self, server_ip):
        """Starts the communication thread.

        Args:
            server_ip (tuple): IP address of the server.

        Examples:
            >>> process.start(('127.0.0.1', 8000))
        """
        self.sub_p = Thread(target=tcp_process, args=(
            (self.out_pipe, self.in_pipe), self.socket_num, server_ip))
        self.sub_p.start()

    def join(self):
        """Waits for the thread to finish execution and closes pipes.

        Examples:
            >>> process.join()
        """
        self.sub_p.join()
        self.in_pipe.close()
        self.out_pipe.close()

    def get_pipe(self):
        """Gets input and output pipes.

        Returns:
            tuple: The input and output pipeline for the current object.

        Examples:
            >>> out_pipe, in_pipe = process.get_pipe()
        """
        return self.out_pipe, self.in_pipe


def tcp_process(pipe, socket_num, server_ip):
    """Handles the main TCP processing logic.

    A process is responsible for the connection between a server and multiple clients.
    The client is responsible for sending datas, while the server is responsible for receiving datas.
    Each server and client has multiple sockets, each responsible for a single connection.

    Args:
        pipe (tuple): Contains input pipes and output pipes.
        socket_num (int): The number of sockets.
        server_ip (tuple): IP address of the server.

    Examples:
        >>> tcp_process((out_p, in_p), 4, ('127.0.0.1', 8000))
    """
    _out_pipe, _in_pipe = pipe
    server_address, server_port = server_ip
    server = TCPServer(server_address, server_port)
    clients = {}

    server_pipes_out = []
    server_pipes_in = []
    server_threads = []
    for i in range(socket_num):
        server.socket_mapping.append({})
        start_thread(server, i, server_pipes_out, server_pipes_in, server_threads)
        time.sleep(3)  # wait the server to activate

    client_pipes_out = {}
    client_pipes_in = {}
    client_threads = {}

    while True:
        try:
            msg = _in_pipe.recv()
            if msg == 'CONNECT':
                other_address, socket_address = _in_pipe.recv()
                client = TCPClient(socket_address[0], socket_address[1])
                clients[str(other_address)] = client
                for i in range(socket_num):
                    server_pipes_in[i].send('CONNECT')
                    if client_pipes_out.get(str(other_address)) is None:
                        client_pipes_out[str(other_address)] = []
                        client_pipes_in[str(other_address)] = []
                        client_threads[str(other_address)] = []
                    client.connect_to_with_retry(other_address[0], other_address[1], i)
                    start_thread(client, i, client_pipes_out[str(other_address)], client_pipes_in[str(other_address)],
                                 client_threads[str(other_address)])
                    time.sleep(1)
                _in_pipe.send('OK')

            elif msg == 'CHECK':
                _in_pipe.send(server.connect_number)

            elif msg == 'SEND':
                target_address = _in_pipe.recv()
                data = _in_pipe.recv()

                tcp = clients.get(str(target_address))

                message_size = struct.pack("Q", len(data))
                send_binary_data(tcp.client_socket[0], message_size)

                data_size = len(data) // socket_num

                for i in range(socket_num):
                    client_pipes_in[str(target_address)][i].send('SEND')
                    client_pipes_in[str(target_address)][i].send(
                        data[i * data_size: (i + 1) * data_size] if i != socket_num - 1 else
                        data[i * data_size:])

                for i in range(socket_num):
                    client_pipes_in[str(target_address)][i].recv()

            elif msg == 'RECV':
                target_address = _in_pipe.recv()
                target_socket = server.socket_mapping[0].get(str(target_address))
                size_data = receive_binary_data(target_socket, struct.calcsize("Q"))
                pure_data_size = struct.unpack("Q", size_data)[0]

                data_size = pure_data_size // socket_num
                last_data_size = pure_data_size - data_size * (socket_num - 1)

                address, port = target_address
                msg_dict = {}
                for i in range(socket_num):
                    target_socket = server.socket_mapping[i].get(str((address, port + i)))
                    server_pipes_in[i].send('RECV')
                    server_pipes_in[i].send(target_socket)
                    server_pipes_in[i].send(data_size if i != socket_num - 1 else last_data_size)

                for i in range(socket_num):
                    msg_dict[i] = server_pipes_in[i].recv()
                data_placeholder = b''.join([msg_dict[i] for i in range(socket_num)])

                _in_pipe.send(data_placeholder)

            elif msg == 'CLOSE':
                for i in range(socket_num):
                    server_pipes_in[i].send('CLOSE')
                    server_threads[i].join()
                    server_pipes_in[i].close()
                    server_pipes_out[i].close()
                    for target, client in clients.items():
                        client_pipes_in[target][i].send('CLOSE')
                        client_threads[target][i].join()
                        client_pipes_in[target][i].close()
                        client_pipes_out[target][i].close()
                break
        except EOFError:
            break


def tcp_thread(pipe, tcp, idx):
    """Implements a thread that handles TCP connections.

    * There are four possible messages to receive:
        * If the *CONNECT* message is received, start TCP server listening;
        * If a *SEND* message is received, the data is received from the pipeline and sent to the specified client using the *send_data* function.
        * If a *RECV* message is received, receive the target socket and data size from the pipeline, use the *receive_data* function to receive the data, and then send it back to the pipeline.
        * If the *CLOSE* message is received, the loop exits, terminating the thread.

    Args:
        pipe (tuple): A tuple containing two pipes (_out_pipe and _in_pipe).
        tcp (Union[TCPServer, TCPClient]): A TCP server or client object.
        idx (int): Identifies a specific client connection.

    Raises:
        EOFError: Exit the loop when the pipe is closed.

    Examples:
        >>> tcp_thread((out_p, in_p), tcp_server, 0)
    """
    _out_pipe, _in_pipe = pipe

    while True:
        try:
            msg = _out_pipe.recv()
            if msg == 'CONNECT':
                assert isinstance(tcp, TCPServer)
                tcp.start_listening(idx)
            elif msg == 'SEND':
                assert isinstance(tcp, TCPClient)
                data = _out_pipe.recv()
                send_data(tcp.client_socket[idx], data)
                _out_pipe.send('OK')
            elif msg == 'RECV':
                assert isinstance(tcp, TCPServer)
                target_socket = _out_pipe.recv()
                data_size = _out_pipe.recv()
                data = receive_data(target_socket, data_size)
                _out_pipe.send(data)
            elif msg == 'CLOSE':
                break
        except EOFError:
            break


def start_thread(tcp, idx, outs, ins, threads):
    """Starts a new thread to handle the TCP connection.

    Args:
        tcp (class): An object instance of a TCP connection.
        idx (int): Used to help manage and distinguish multiple concurrent TCP connections.
        outs (list): The sending end list of the pipeline.
        ins (list): The receiving end list of the pipeline.
        threads (list): The started thread list.

    Examples:
        >>> start_thread(tcp_server, 0, outs, ins, threads)
    """
    pipe = Pipe(True)
    out, in_ = pipe
    thread = Thread(target=tcp_thread, args=(pipe, tcp, idx))
    thread.start()

    outs.append(out)
    ins.append(in_)
    threads.append(thread)
