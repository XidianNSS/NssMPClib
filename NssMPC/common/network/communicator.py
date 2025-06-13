"""
Communication object
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import pickle

from NssMPC.common.network.tcp_process import TCPProcess
from NssMPC.common.utils import *
from NssMPC.config import SOCKET_NUM, DEVICE
from NssMPC.common.network.async_tcp import TCPServer, TCPClient


class Communicator:
    """
    Provides the most basic communication functions.
    """

    def __init__(self):
        """
        Initializes a Communicator object.

        ATTRIBUTES:
            * **address** (*str*): The address to bind the object to.
            * **port** (*int*): The port to bind the object to.
            * **comm_rounds** (*dict*): Communication frequency.It is divided into receiving information and sending information two cases.
            * **comm_rounds** (*dict*): Communication bytes.It is divided into receiving information and sending information two cases.
            * **max_connections** (*int*): the maximum number of communication connections for the current object.
            * **server_socket** (*socket.socket*): The socket object used for communication.
            * **sending_socket_mapping** (*dict*): A dictionary mapping client addresses to their respective sockets.
        """
        self.address = None
        self.port = None
        self.comm_rounds = {'send': 0, 'recv': 0}
        self.comm_bytes = {'send': 0, 'recv': 0}
        self.max_connections = None
        self.server_socket = None
        self.sending_socket_mapping = {}

    def set_address(self, address, port):
        """
        Set communication address.

        Assigns the passed arguments to the address and port properties of the object.

        :param address: The address to bind the object to.
        :type address: str
        :param port: The port to bind the object to.
        :type port: int
        """
        self.address = address
        self.port = port

    def set_max_connections(self, max_connections):
        """
        Sets the maximum number of communication connections for the current object.

        :param max_connections: the maximum number of communication connections for the current object.
        :type max_connections: int
        """
        self.max_connections = max_connections

    def init_server(self):
        """
        Initialize the server.
        """
        raise NotImplementedError

    def connect_to_other(self, other_address, socket_address):
        """
        Connect via the address of **other**.

        :param other_address: The address of *other* to connect to.
        :type other_address: tuple
        :param socket_address: local socket address.
        :type socket_address: tuple
        """
        raise NotImplementedError

    def connected_num(self):
        """
        Get the number of connected clients.
        """
        raise NotImplementedError

    def _send_to(self, address, x):
        """
        Send **x** to the target address **address**.

        :param address: The target address to send to.
        :type address: tuple
        :param x: Information that needs to be sent.
        :type x: Any

        """
        raise NotImplementedError

    def _recv_from(self, address):
        """
        Receives information from **address**.

        :param address: The address to receive the message.
        :type address: tuple

        """
        raise NotImplementedError

    def _close(self):
        """
        Disconnect from the target object.
        """
        raise NotImplementedError

    def send_to_address(self, address, x):
        """
        Send *x* to the target address *address*.

        Call the function :py:meth:`_send_to` that sends the message.The number of times the message is sent is increased by one,
        and the number of bytes sent is increased by x bytes.

        :param address: The target address to send to.
        :type address: tuple
        :param x: Information that needs to be sent.
        :type x: Any
        """
        self._send_to(address, x)
        self.comm_rounds['send'] += 1
        self.comm_bytes['send'] += count_bytes(x)

    def recv_from_address(self, address):
        """
        Receive message from the address *address*.

        Call the function :py:meth:`_recv_from()` that receives the message.The number of times the message is received is increased by one,
        and the number of bytes receives is increased by item bytes.

        :param address: Sender address.
        :type address: tuple
        """
        item = self._recv_from(address)
        self.comm_rounds['recv'] += 1
        self.comm_bytes['recv'] += count_bytes(item)
        if hasattr(item, 'device'):
            return item.to(DEVICE)
        return item

    def close(self):
        """
        This method causes the communicating party to stop communicating.

        According to calling close function :py:meth:`_close()` to stop communicating, and print the communication shutdown log.
        """
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
        """
        Initializes a NCommunicator object.

        The method of the parent class Communicator is called to ensure that the parent class's
        initialization logic is executed.

        ATTRIBUTES:
            * **tcp_process** (:class:`~NssMPC.common.network.tcp_process.TCPProcess`): Used to manage the processing and communication of TCP connections, enabling the NCommunicator class to transmit data or interact with other network nodes
            * **in_pipe** (*Pipe*): Used to manage the processing and communication of TCP connections, enabling the NCommunicator class to transmit data or interact with other network nodes
            * **out_pipe** (*Pipe*): Used to manage the processing and communication of TCP connections, enabling the NCommunicator class to transmit data or interact with other network nodes
            * **socket_num** (*int*): Used to manage the processing and communication of TCP connections, enabling the NCommunicator class to transmit data or interact with other network nodes
            * **connected_parties** (*int*): Used to manage the processing and communication of TCP connections, enabling the NCommunicator class to transmit data or interact with other network nodes
        """
        super(MCommunicator, self).__init__()
        self.tcp_process = None
        self.in_pipe = None
        self.out_pipe = None
        self.socket_num = SOCKET_NUM
        self.connected_parties = 0

    def set_socket_num(self, socket_num):
        """
        Sets the number of sockets.

        :param socket_num: the number of sockets.
        :type socket_num: int
        """
        self.socket_num = socket_num

    def init_server(self):
        """
        This method initializes the server to facilitate subsequent communication.
        According to passing a tuple as an argument to create an instance of the TCPProcess class.And get
        *self.out_pipe* and *self.in_pipe* through the :meth:`get_pipe <NssMPC.common.network.tcp_process.TCPProcess.get_pipe>` method of *tcp_process*.
        Finally, the :meth:`~NssMPC.common.network.tcp_process.TCPProcess.start` method of the *tcp_process* is called to start the communication thread.
        """
        self.tcp_process = TCPProcess(server_ip=(self.address, self.port))
        self.out_pipe, self.in_pipe = self.tcp_process.get_pipe()
        self.tcp_process.start((self.address, self.port))

    def connect_to_other(self, other_address, socket_address):
        """
        This method connects the local socket address to the destination socket address.

        The output pipe first issues a send request until it receives an OK reply.

        :param other_address: The address of *other* to connect to.
        :type other_address: tuple
        :param socket_address:  local socket address.
        :type socket_address: tuple
        :raises ConnectionError: If local socket address do not receive OK reply.
        """
        self.out_pipe.send('CONNECT')
        self.out_pipe.send((other_address, socket_address))
        if self.out_pipe.recv() == 'OK':
            print("All sockets are connected to", other_address)
        else:
            raise ConnectionError("Failed to connect.")

    def connected_num(self):
        """
        Check the status of connections and get the number or information of connections.

        The output pipe first send *CHECK* to each connected address, then receives the response and returns it.

        :return: The response received by the out pipe.
        :rtype: Any
        """
        self.out_pipe.send('CHECK')
        return self.out_pipe.recv()

    def _send_to(self, address, x):
        """
        Sending data to a specified address.

        Firstly, the local socket address sends a *SEND* message to indicate that the next data will be sent.Then
        serializes data x into byte format using *pickle.dumps(x)* for easy transfer.

        :param address: The destination address for receiving the message
        :type address: tuple
        :param x: The massage ready to be sent.
        :type x: Any

        """
        self.out_pipe.send('SEND')
        self.out_pipe.send(address)
        msg = pickle.dumps(x)
        self.out_pipe.send(msg)

    def _recv_from(self, address):
        """
        Receiving data from a specified address.

        Firstly, the local socket address sends a *RECV* message to indicate that the next data will be received.Then
        uses method *recv()* to receive message.Finally, Deserialization of serialized byte data msg into raw Python objects with *pickle.loads(msg)*.

        :param address: The destination address for receiving the message
        :type address: tuple
        :return: Received information
        :rtype: Any

        """
        self.out_pipe.send('RECV')
        self.out_pipe.send(address)
        msg = self.out_pipe.recv()
        msg = pickle.loads(msg)
        return msg

    def _close(self):
        """
        Disable operations related to connections.

        The output pipe sends a *CLOSE* message to notify other parts to stop operations or close connections. Call
        *join()*, which blocks the current thread until the tcp_process thread or process completes,
        ensuring that all resources are properly released.
        """
        self.out_pipe.send('CLOSE')
        self.tcp_process.join()


class NCommunicator(Communicator):
    def __init__(self):
        """
        The method of the parent class Communicator is called to ensure that the parent class's
        initialization logic is executed.
        """
        super(NCommunicator, self).__init__()

    def connect_to_other(self, other_address, socket_address):
        """
        Connect via the address of **other**.

        First create a communication client, pass in the client's address and port, then call :meth:`~NssMPC.common.network.async_tcp.TCPClient.connect_to_with_retry`
        to connect to the target server, and finally map the successful connection to the current object's dictionary

        :param other_address: target address to connect.
        :type other_address: tuple
        :param socket_address: local address.
        :type socket_address: tuple
        """
        client_socket = TCPClient(socket_address[0], socket_address[1])
        client_socket.connect_to_with_retry(other_address[0], other_address[1])
        self.sending_socket_mapping[str(other_address)] = client_socket

    def init_server(self):
        """
        Initializing the communication server.This method first creates a server socket through the address and port
        properties of the object, and then executes the **run()** method of the server_socket is called to start the
        communication thread.
        """
        self.server_socket = TCPServer(self.address, self.port)
        self.server_socket.run()

    def connected_num(self):
        """
        Gets the number of clients connected to the server.

        :return: the number of clients connected to the server
        :rtype: int
        """
        return self.server_socket.connect_number

    def _recv_from(self, address):
        """
        Call the :meth:`~NssMPC.common.network.async_tcp.TCPServer.receive_serializable_item_from` method of the server_socket attribute to receive data from the target address

        :param address: target address to communicate.
        :type address:tuple
        :return: Received message
        :rtype: Any

        """
        item = self.server_socket.receive_serializable_item_from(str(address))
        return item

    def _send_to(self, address, x):
        """
        Send **x** to the target address **address**.

        Firstly, convert address to a string type,then call the :meth:`~NssMPC.common.network.async_tcp.TCPClient.send_serializable_item`
        method of the socket attribute to send an object that can be serialized.

        :param address: The target address to send to.
        :type address: tuple
        :param x: Information that needs to be sent.
        :type x: Any

        """
        socket = self.sending_socket_mapping.get(str(address))
        socket.send_serializable_item(x)

    def _close(self):
        """
        This method causes the communicating party to stop communicating.

        Obtain the addresses and ports of all connected clients from sending_socket_mapping and disconnect them one by one.

        .. note::
            The difference between the two close() is:
                MCommunicator is waiting for tcpprocess to end, requiring interprocess collaboration.
                NCommunicator focuses on closing all sockets.
        """
        for address, socket in self.sending_socket_mapping.items():
            socket.close()
        self.server_socket.close_all()
