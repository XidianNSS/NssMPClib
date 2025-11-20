"""
Communication object
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import pickle

from NssMPC.config import SOCKET_NUM, DEVICE
from NssMPC.infra.mpc.communication.async_tcp import TCPServer, TCPClient
from NssMPC.infra.mpc.communication.tcp_process import TCPProcess
from NssMPC.infra.utils.debug_utils import count_bytes, bytes_convert


class Communicator:
    """
    Provides the most basic communication functions.

    Attributes:
        address (str): The address to bind the object to.
        port (int): The port to bind the object to.
        comm_rounds (dict): Communication frequency. It is divided into receiving information and sending information two cases.
        comm_bytes (dict): Communication bytes. It is divided into receiving information and sending information two cases.
        max_connections (int): The maximum number of communication connections for the current object.
        server_socket (socket.socket): The socket object used for communication.
        sending_socket_mapping (dict): A dictionary mapping client addresses to their respective sockets.
    """

    def __init__(self):
        """
        Initializes a Communicator object.

        Examples:
            >>> communicator = Communicator()
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

        Args:
            address (str): The address to bind the object to.
            port (int): The port to bind the object to.

        Examples:
            >>> communicator.set_address('127.0.0.1', 8000)
        """
        self.address = address
        self.port = port

    def set_max_connections(self, max_connections):
        """
        Sets the maximum number of communication connections for the current object.

        Args:
            max_connections (int): The maximum number of communication connections for the current object.

        Examples:
            >>> communicator.set_max_connections(10)
        """
        self.max_connections = max_connections

    def init_server(self):
        """
        Initialize the server.

        Raises:
            NotImplementedError: This method is abstract.

        Examples:
            >>> communicator.init_server()
        """
        raise NotImplementedError

    def connect_to_other(self, other_address, socket_address):
        """
        Connect via the address of other.

        Args:
            other_address (tuple): The address of other to connect to.
            socket_address (tuple): Local socket address.

        Raises:
            NotImplementedError: This method is abstract.

        Examples:
            >>> communicator.connect_to_other(('127.0.0.1', 8001), ('127.0.0.1', 8000))
        """
        raise NotImplementedError

    def connected_num(self):
        """
        Get the number of connected clients.

        Raises:
            NotImplementedError: This method is abstract.

        Examples:
            >>> num = communicator.connected_num()
        """
        raise NotImplementedError

    def _send_to(self, address, x):
        """
        Send x to the target address.

        Args:
            address (tuple): The target address to send to.
            x (Any): Information that needs to be sent.

        Raises:
            NotImplementedError: This method is abstract.

        Examples:
            >>> communicator._send_to(('127.0.0.1', 8001), "data")
        """
        raise NotImplementedError

    def _recv_from(self, address):
        """
        Receives information from address.

        Args:
            address (tuple): The address to receive the message.

        Raises:
            NotImplementedError: This method is abstract.

        Examples:
            >>> data = communicator._recv_from(('127.0.0.1', 8001))
        """
        raise NotImplementedError

    def _close(self):
        """
        Disconnect from the target object.

        Raises:
            NotImplementedError: This method is abstract.

        Examples:
            >>> communicator._close()
        """
        raise NotImplementedError

    def send_to_address(self, address, x):
        """
        Send x to the target address.

        Call the function `_send_to` that sends the message. The number of times the message is sent is increased by one,
        and the number of bytes sent is increased by x bytes.

        Args:
            address (tuple): The target address to send to.
            x (Any): Information that needs to be sent.

        Examples:
            >>> communicator.send_to_address(('127.0.0.1', 8001), "data")
        """
        self._send_to(address, x)
        self.comm_rounds['send'] += 1
        self.comm_bytes['send'] += count_bytes(x)

    def recv_from_address(self, address):
        """
        Receive message from the address.

        Call the function `_recv_from()` that receives the message. The number of times the message is received is increased by one,
        and the number of bytes receives is increased by item bytes.

        Args:
            address (tuple): Sender address.

        Returns:
            Any: The received item.

        Examples:
            >>> data = communicator.recv_from_address(('127.0.0.1', 8001))
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

        According to calling close function `_close()` to stop communicating, and print the communication shutdown log.

        Examples:
            >>> communicator.close()
        """
        self._close()
        print(f"Communicator: {self.address} closed.")
        print(f"Communication costs:\n\tsend rounds: {self.comm_rounds['send']}\t\t"
              f"send bytes: {bytes_convert(self.comm_bytes['send'])}.")
        print(f"\trecv rounds: {self.comm_rounds['recv']}\t\t"
              f"recv bytes: {bytes_convert(self.comm_bytes['recv'])}.")


class MCommunicator(Communicator):
    """
    Multi-socket Communicator.

    Attributes:
        tcp_process (TCPProcess): Used to manage the processing and communication of TCP connections.
        in_pipe (Pipe): Input pipe for communication.
        out_pipe (Pipe): Output pipe for communication.
        socket_num (int): Number of sockets.
        connected_parties (int): Number of connected parties.
    """

    def __init__(self):
        """
        Initializes a MCommunicator object.

        The method of the parent class Communicator is called to ensure that the parent class's
        initialization logic is executed.

        Examples:
            >>> m_comm = MCommunicator()
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

        Args:
            socket_num (int): The number of sockets.

        Examples:
            >>> m_comm.set_socket_num(4)
        """
        self.socket_num = socket_num

    def init_server(self):
        """
        This method initializes the server to facilitate subsequent communication.

        According to passing a tuple as an argument to create an instance of the TCPProcess class. And get
        `self.out_pipe` and `self.in_pipe` through the `get_pipe` method of `tcp_process`.
        Finally, the `start` method of the `tcp_process` is called to start the communication thread.

        Examples:
            >>> m_comm.init_server()
        """
        self.tcp_process = TCPProcess(server_ip=(self.address, self.port))
        self.out_pipe, self.in_pipe = self.tcp_process.get_pipe()
        self.tcp_process.start((self.address, self.port))

    def connect_to_other(self, other_address, socket_address):
        """
        This method connects the local socket address to the destination socket address.

        The output pipe first issues a send request until it receives an OK reply.

        Args:
            other_address (tuple): The address of other to connect to.
            socket_address (tuple): Local socket address.

        Raises:
            ConnectionError: If local socket address do not receive OK reply.

        Examples:
            >>> m_comm.connect_to_other(('127.0.0.1', 8001), ('127.0.0.1', 8000))
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

        The output pipe first send CHECK to each connected address, then receives the response and returns it.

        Returns:
            Any: The response received by the out pipe.

        Examples:
            >>> num = m_comm.connected_num()
        """
        self.out_pipe.send('CHECK')
        return self.out_pipe.recv()

    def _send_to(self, address, x):
        """
        Sending data to a specified address.

        Firstly, the local socket address sends a SEND message to indicate that the next data will be sent. Then
        serializes data x into byte format using `pickle.dumps(x)` for easy transfer.

        Args:
            address (tuple): The destination address for receiving the message.
            x (Any): The message ready to be sent.

        Examples:
            >>> m_comm._send_to(('127.0.0.1', 8001), "data")
        """
        self.out_pipe.send('SEND')
        self.out_pipe.send(address)
        msg = pickle.dumps(x)
        self.out_pipe.send(msg)

    def _recv_from(self, address):
        """
        Receiving data from a specified address.

        Firstly, the local socket address sends a RECV message to indicate that the next data will be received. Then
        uses method `recv()` to receive message. Finally, Deserialization of serialized byte data msg into raw Python objects with `pickle.loads(msg)`.

        Args:
            address (tuple): The destination address for receiving the message.

        Returns:
            Any: Received information.

        Examples:
            >>> data = m_comm._recv_from(('127.0.0.1', 8001))
        """
        self.out_pipe.send('RECV')
        self.out_pipe.send(address)
        msg = self.out_pipe.recv()
        msg = pickle.loads(msg)
        return msg

    def _close(self):
        """
        Disable operations related to connections.

        The output pipe sends a CLOSE message to notify other parts to stop operations or close connections. Call
        `join()`, which blocks the current thread until the tcp_process thread or process completes,
        ensuring that all resources are properly released.

        Examples:
            >>> m_comm._close()
        """
        self.out_pipe.send('CLOSE')
        self.tcp_process.join()


class NCommunicator(Communicator):
    def __init__(self):
        """
        Initializes a NCommunicator object.

        The method of the parent class Communicator is called to ensure that the parent class's
        initialization logic is executed.

        Examples:
            >>> n_comm = NCommunicator()
        """
        super(NCommunicator, self).__init__()

    def connect_to_other(self, other_address, socket_address):
        """
        Connect via the address of other.

        First create a communication client, pass in the client's address and port, then call `TCPClient.connect_to_with_retry`
        to connect to the target server, and finally map the successful connection to the current object's dictionary.

        Args:
            other_address (tuple): Target address to connect.
            socket_address (tuple): Local address.

        Examples:
            >>> n_comm.connect_to_other(('127.0.0.1', 8001), ('127.0.0.1', 8000))
        """
        client_socket = TCPClient(socket_address[0], socket_address[1])
        client_socket.connect_to_with_retry(other_address[0], other_address[1])
        self.sending_socket_mapping[str(other_address)] = client_socket

    def init_server(self):
        """
        Initializing the communication server.

        This method first creates a server socket through the address and port
        properties of the object, and then executes the `run()` method of the server_socket is called to start the
        communication thread.

        Examples:
            >>> n_comm.init_server()
        """
        self.server_socket = TCPServer(self.address, self.port)
        self.server_socket.run()

    def connected_num(self):
        """
        Gets the number of clients connected to the server.

        Returns:
            int: The number of clients connected to the server.

        Examples:
            >>> num = n_comm.connected_num()
        """
        return self.server_socket.connect_number

    def _recv_from(self, address):
        """
        Call the `TCPServer.receive_serializable_item_from` method of the server_socket attribute to receive data from the target address.

        Args:
            address (tuple): Target address to communicate.

        Returns:
            Any: Received message.

        Examples:
            >>> data = n_comm._recv_from(('127.0.0.1', 8001))
        """
        item = self.server_socket.receive_serializable_item_from(str(address))
        return item

    def _send_to(self, address, x):
        """
        Send x to the target address.

        Firstly, convert address to a string type, then call the `TCPClient.send_serializable_item`
        method of the socket attribute to send an object that can be serialized.

        Args:
            address (tuple): The target address to send to.
            x (Any): Information that needs to be sent.

        Examples:
            >>> n_comm._send_to(('127.0.0.1', 8001), "data")
        """
        socket = self.sending_socket_mapping.get(str(address))
        socket.send_serializable_item(x)

    def _close(self):
        """
        This method causes the communicating party to stop communicating.

        Obtain the addresses and ports of all connected clients from sending_socket_mapping and disconnect them one by one.

        Note:
            The difference between the two close() is:
                MCommunicator is waiting for tcpprocess to end, requiring interprocess collaboration.
                NCommunicator focuses on closing all sockets.

        Examples:
            >>> n_comm._close()
        """
        for address, socket in self.sending_socket_mapping.items():
            socket.close()
        self.server_socket.close_all()
