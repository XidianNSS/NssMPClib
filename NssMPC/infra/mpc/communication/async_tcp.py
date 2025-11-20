"""
Async Transmission Control Protocol
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import pickle
import socket
import struct
import threading
import time

from NssMPC.config import SOCKET_MAX_SIZE


class TCPServer(object):
    """
    A simple TCP server that listens for incoming connections and maintains client connections.
    Sends and receives data objects over a communication socket.

    Attributes:
        address (tuple): A tuple containing the address and port for the server.
        server_socket (socket.socket): The socket object used for communication.
        socket_mapping (dict): A dictionary mapping client addresses to their respective sockets.
        connect_number (int): The number of currently connected clients.
        close_tag (bool): Used to mark whether to shut down the server.
        listening_thread (threading.Thread): The thread that handles listening for incoming connection.
    """

    def __init__(self, address, port):
        """
        Initializes the TCPServer with a specific address and port.

        Args:
            address (str): The IP address for the server.
            port (int): The port number for the server.

        Examples:
            >>> server = TCPServer('127.0.0.1', 8080)
        """
        self.address = (address, port)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(self.address)
        self.server_socket.listen(5)
        self.socket_mapping = {}
        self.connect_number = 0
        self.close_tag = False
        self.listening_thread = None

    def run(self):
        """
        Starts the server in a separate thread to handle incoming connections.

        Examples:
            >>> server.run()
        """
        self.listening_thread = threading.Thread(target=self.start_listening)
        self.listening_thread.start()

    def start_listening(self):
        """
        Set the server to start listening.

        This method makes the server continue listening for client connections
        until `self.close_tag` is set to True and receive a disconnection indication.

        Note:
            The timeout duration of the server socket is 5s. If the timeout occurs, the `self.close_tag` exception is displayed.

        If the connection was successful:
            A new socket object is returned with the client's address printed indicating that the
            connection was successful. Store client sockets in `socket_mapping` (a dictionary for storing client sockets,
            with the client address as the key) and add `connect_number` (the number of connections to the server) by one.

        Examples:
            >>> server.start_listening()
        """
        print("TCPServer waiting for connection ......")
        while True:
            self.server_socket.settimeout(5)
            if self.close_tag:
                break
            try:
                client_socket, client_address = self.server_socket.accept()
                print("TCPServer successfully connected by :%s" % str(client_address))
                self.socket_mapping[str(client_address)] = client_socket
                self.connect_number += 1
            except socket.timeout:
                continue

    def receive_serializable_item_from(self, target_address):
        """
        This method lets the server receive serialized data objects from a socket at the specified destination address.

        The server first calculates the number of bytes occupied by the received data object, and gets the actual
        size of the data by decoding, and then reads the data byte by byte. And then we're going to deserialize
        `data_placeholder` to a Python object `frame`.

        Note:
            "Q" represents an unsigned 8-byte integer, Therefore, struct.calcsize("Q") will return 8.

        Args:
            target_address (str): A str containing the address and port for the target.

        Returns:
            Any: The resulting object `frame` deserialized.

        Examples:
            >>> data = server.receive_serializable_item_from('127.0.0.1:12345')
        """
        target_socket = self.socket_mapping.get(target_address)
        size_data = receive_binary_data(target_socket, struct.calcsize("Q"))
        pure_data_size = struct.unpack("Q", size_data)[0]
        data_placeholder = bytearray(pure_data_size)
        offset = 0
        while offset < pure_data_size:
            remain_size = pure_data_size - offset
            next_receive_size = min(remain_size, SOCKET_MAX_SIZE)
            new_data = receive_binary_data(target_socket, next_receive_size)
            data_placeholder[offset: offset + len(new_data)] = new_data
            offset += len(new_data)
        frame = pickle.loads(data_placeholder)
        return frame

    def close(self, target_address):
        """
        Disconnect from the target client.

        After receiving an indication to disconnect from the target address, delete the key-value pair corresponding
        to target_address, remove the record for that connection, and subtract by one the number of clients connected
        to itself.

        Args:
            target_address (str): A tuple containing the address and port for the target.

        Examples:
            >>> server.close('127.0.0.1:12345')
        """
        self.socket_mapping[target_address].detach()
        self.socket_mapping[target_address].close()
        self.socket_mapping.pop(target_address)
        self.connect_number -= 1

    def close_all(self):
        """
        Close all server-related resources and connections.

        First, iterate over each connected address and call the `close()` method to close the socket connection. To
        clear the number of connections and set `self.close_tag` to True, the server is shut down, close the main
        socket of the server after the listening thread ends, and stop the server from accepting connections.

        Examples:
            >>> server.close_all()
        """
        for address in self.socket_mapping:
            self.socket_mapping[address].detach()
            self.socket_mapping[address].close()
        self.socket_mapping.clear()
        self.connect_number = 0
        # shut down server
        self.close_tag = True
        if self.listening_thread is not None:
            self.listening_thread.join()
        try:
            self.server_socket.detach()
        except Exception:
            pass
        self.server_socket.close()


class TCPClient(object):
    """
    A simple TCP Client that listens for incoming connections and maintains server connections.
    Sends and receives data objects over a communication socket.

    Attributes:
        target_address (tuple): A tuple containing the host and port for the target party.
        self_address (tuple): A tuple containing the self_address and self_port for the self party.
        client_socket (socket.socket): A TCP socket object created by the client, used to establish a connection to the server.
    """

    def __init__(self, self_address, self_port):
        """
        Initializes the TCPClient.

        Args:
            self_address (str): The IP address of the client.
            self_port (int): The port number of the client.

        Examples:
            >>> client = TCPClient('127.0.0.1', 9090)
        """
        self.target_address = None
        self.self_address = (self_address, self_port)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.client_socket.bind((self_address, self_port))

    def connect_to(self, host, port):
        """
        Connect to the specified address and port of the target server.

        Args:
            host (str): Destination server address.
            port (int): Destination server port.

        Examples:
            >>> client.connect_to('192.168.1.100', 8080)
        """
        self.target_address = (host, port)
        self.client_socket.connect(self.target_address)
        print("successfully connect to server %s:%d" % (host, port))

    def connect_to_with_retry(self, host, port):
        """
        Connect to the target server with retry mechanism.

        If the connection is successful, exit the loop. If the connection fails, catch the connection rejection exception, pause for 1s and try again.

        Args:
            host (str): Destination server address.
            port (int): Destination server port.

        Examples:
            >>> client.connect_to_with_retry('192.168.1.100', 8080)
        """
        while True:
            try:
                self.connect_to(host, port)
                break
            except ConnectionRefusedError:
                print('failed to connect to server: %s:%d' % (host, port))
                time.sleep(1)
                continue

    def send_msg(self, msg):
        """
        Send ordinary message.

        Args:
            msg (bytes): The message ready to send.

        Examples:
            >>> client.send_msg(b'Hello World')
        """
        send_binary_data(self.client_socket, msg)

    def send_serializable_item(self, item):
        """
        Send an object that can be serialized.

        First, the object is serialized into a byte stream, and the length of the serialized data is packaged into an
        8-byte unsigned long integer before the length of the data is sent. The data is then sent in a loop until all
        data is sent.

        * When sending data in a loop:
            1. If the length of the remaining data is less than the maximum send size, the remaining data is sent.
            2. If the remaining data length is greater than the maximum send size, the data is sent in blocks.

        Args:
            item (Any): The object to be sent.

        Examples:
            >>> client.send_serializable_item({'key': 'value'})
        """
        data = pickle.dumps(item)
        message_size = struct.pack("Q", len(data))
        send_binary_data(self.client_socket, message_size)

        send_length = 0
        while send_length < len(data):
            if send_length < SOCKET_MAX_SIZE:
                send_size = self.client_socket.send(data[send_length:])
            else:
                send_size = self.client_socket.send(data[send_length: send_length + SOCKET_MAX_SIZE])
            send_length += send_size

    def close(self):
        """
        Close the client socket.

        Examples:
            >>> client.close()
        """
        self.client_socket.detach()
        self.client_socket.close()


def send_binary_data(__socket, data):
    """
    Helper function for sending binary data.

    Args:
        __socket (socket.socket): The socket for sending messages.
        data (bytes): The binary data ready to send.

    Examples:
        >>> send_binary_data(sock, b'data')
    """
    __socket.send(data)


def receive_binary_data(__socket, msg_size):
    """
    An auxiliary function that receives binary data.

    Args:
        __socket (socket.socket): The socket for receiving messages.
        msg_size (int): Byte size.

    Returns:
        bytes: Receives `msg_size` bytes of data from the specified socket.

    Examples:
        >>> data = receive_binary_data(sock, 1024)
    """
    return __socket.recv(msg_size)
