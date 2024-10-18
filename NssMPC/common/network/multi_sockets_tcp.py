"""
TCP server and client related framework
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import socket
import struct
import threading
import time

from NssMPC.config import SOCKET_MAX_SIZE, SOCKET_NUM


class TCPServer(object):
    """
    The class is a basic framework for creating multithreaded TCP servers.
    """

    def __init__(self, address, port):
        """
        Example Initialize the server address and port.

        ATTRIBUTES:
            * **address** (*str*): The address to bind the object to.
            * **port** (*int*): The port to bind the object to.
            * **server_socket** (*list*): A list used to store multiple server sockets.
            * **socket_mapping** (*list*): Stores a dictionary list of client sockets corresponding to each server socket.
            * **connect_number** (*int*): Records the number of currently connected clients
            * **close_tag** (*bool*): Used to mark whether the server should be shut down.
            * **listening_thread** (*Optional[threading.Thread]*): Stores a reference to the listening thread
            * **socket_num** (*int*): The number of server sockets to create.
        """
        self.address = (address, port)
        self.server_socket = []
        self.socket_mapping = []
        self.connect_number = 0
        self.close_tag = False
        self.listening_thread = None
        self.socket_num = SOCKET_NUM
        self.start()

    def start(self):
        """
        Start connection.

        Firstly, Create a specified number of server sockets, then set the reuse address option, bind the address and port, and start listening for connections

        .. note::
            * **Socket. AF_INET:** Specifies that the socket uses the IPv4 address protocol
            * **Socket.sock_stream:** Specifies the socket as a stream socket for TCP connections
            * The current socket is set to listening mode and can queue up to 5 connection requests
        """
        address, port = self.address
        for i in range(self.socket_num):
            self.server_socket.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            # setsockopt sets the socket options. Socket.sol_socket: Specifies the option applied to the socket itself. SO_REUSEADDR allows you to rebind an address and port in use
            self.server_socket[i].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket[i].bind(
                (address, port + i))  # The port is port+i, in order to bind each socket to a different port
            self.server_socket[i].listen(5)
            self.socket_mapping.append({})

    def run(self):
        """
        Start a listening thread.

        Create a new thread for each server socket to handle connection requests.
        """
        for i in range(self.socket_num):
            self.socket_mapping.append({})
            self.listening_thread = threading.Thread(target=self.start_listening, args=(i,))
            self.listening_thread.start()

    def start_listening(self, server_idx):
        """
        Wait for a connection from the client.

        By setting the timeout time of the socket, to accept the connection, and logs the client information.
            * If the connection is successful, the client socket is saved to socket_mapping, and update the connect_number.
            * If the connection fails, jump out of the loop and enter the next round of waiting.

        :param server_idx: index of the server
        :type server_idx: int
        """
        print(f"TCPServer{server_idx} waiting for connection ......")
        while True:
            self.server_socket[server_idx].settimeout(5)
            try:
                client_socket, client_address = self.server_socket[server_idx].accept()
                print(f"TCPServer{server_idx} successfully connected by :%s" % str(client_address))
                self.socket_mapping[server_idx][str(client_address)] = client_socket
                self.connect_number += 1
                break
            except socket.timeout:
                continue


class TCPClient(object):
    """
    The class is a basic framework for creating multithreaded TCP clients.
    """

    def __init__(self, self_address, self_port):
        """
        Example Initialize the client address and port.

        * **target_address** (*list*): Stores the address list of the target server
        * **self_address** (*tuple*): Local client address
        * **client_socket** (*socket.socket*):  A TCP socket object created by the client. This object is used to establish a connection to the server.
        * **server_socket** (*socket.socket*): A TCP socket object created by the server.
        * **payload_size** (*int*): Calculates the number of load bytes.
        * **data** (*int*): Stores sent or received data, initially as an empty byte string
        * **send_lock** (*threading.Lock*): Thread lock. Ensure thread-safe sending operations
        * **socket_num** (*int*): Number of client sockets created
        """
        self.target_address = []
        self.self_address = (self_address, self_port)
        self.client_socket = []
        self.server_socket = None
        self.payload_size = struct.calcsize("Q")
        self.data = b''
        self.send_lock = threading.Lock()
        self.socket_num = SOCKET_NUM
        self.start()

    def start(self):
        """
        Start connection.

        Firstly, set the reuse address and keep-connected options, then, bind to different ports for concurrent use.
        """
        address, port = self.self_address
        for i in range(self.socket_num):
            self.client_socket.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            self.client_socket[i].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.client_socket[i].setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.client_socket[i].bind((address, port + i))

    def connect_to(self, host, port, idx):
        """
        Connect to the specified server address and port.

        First, the destination address and port are added to the list, and then the local communication address begins to connect

        :param host: The host name or IP address of the target server
        :type host: str
        :param port: Port number of the target server.
        :type port: int
        :param idx: Identifies the client socket that is being connected.
        :type idx: int
        :return: The status code of the connection result
        :rtype: int
        """
        self.target_address.append((host, port + idx))
        return self.client_socket[idx].connect_ex(self.target_address[idx])

    def connect_to_with_retry(self, host, port, idx):
        """
        Loop through attempts to connect to the target server until the connection is successful.

        If the connection fails, an error message is printed and tried again every second until successful.

        :param host: The host name or IP address of the target server
        :type host: str
        :param port: Port number of the target server.
        :type port: int
        :param idx: Identifies the client socket that is being connected.
        :type idx: int
        """
        while True:
            flag = self.connect_to(host, port, idx)
            if flag == 0:
                print(f"successfully connect to server {host}:{port + idx}")
                break
            else:
                print('failed to connect to server: %s:%d' % (host, port + idx))
                time.sleep(1)
                continue


def send_binary_data(__socket, data):
    """
    Sends binary data to the specified socket.

    :param __socket: Socket object
    :type __socket: socket.socket
    :param data: Binary data that needs to be sent
    :type data: Any
    """
    __socket.send(data)


def receive_binary_data(__socket, msg_size):
    """
    Receives binary data of a specified size from a specified socket

    :param __socket: Socket object
    :type __socket: socket.socket
    :param msg_size: Size of data to receive (bytes)
    :type msg_size: int
    :return: Received data
    :rtype: Any
    """
    return __socket.recv(msg_size)


def receive_data(__socket, data_size):
    """
    Receives complete data from the socket until the specified number of bytes is received.

    When the offset does not reach the required data size, the data is continuously received through a loop. The min function is used to determine the size of the received data this time, ensuring that the size does not exceed *SOCKET_MAX_SIZE*. The offset is updated after the received data is stored.

    :param __socket: Socket object
    :type __socket: socket.socket
    :param data_size: Size of data to receive (bytes)
    :type data_size: int
    :return: Received data
    :rtype: list
    """
    data_placeholder = bytearray(data_size)
    offset = 0
    while offset < data_size:
        remain_size = data_size - offset
        next_receive_size = min(remain_size, SOCKET_MAX_SIZE)
        new_data = receive_binary_data(__socket, next_receive_size)
        data_placeholder[offset: offset + len(new_data)] = new_data
        offset += len(new_data)
    return data_placeholder


def send_data(__socket, msg):
    """
    Send complete data to the socket.

    In this method ,data is continuously sent when the sending length is less than the message length. Firstly,check the remaining bytes to be sent to determine the size of the current send.
    If the remaining data is less than the maximum transfer size *SOCKET_MAX_SIZE*, the remaining part is sent. Otherwise, it sends a block of maximum length.

    :param __socket: Socket object
    :type __socket: socket.socket
    :param msg: Massage needed to be sent
    :type msg: Any
    """
    send_length = 0
    while send_length < len(msg):
        if len(msg) - send_length < SOCKET_MAX_SIZE:
            send_size = __socket.send(msg[send_length:])
        else:
            send_size = __socket.send(msg[send_length: send_length + SOCKET_MAX_SIZE])
        send_length += send_size
