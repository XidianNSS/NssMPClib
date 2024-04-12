import pickle
import socket
import struct
import threading
import time

from config.network_configs import SOCKET_MAX_SIZE


class TCPServer(object):
    def __init__(self, address, port):
        self.address = (address, port)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(self.address)
        self.server_socket.listen(5)
        self.socket_mapping = {}
        self.connect_number = 0
        self.close_tag = None
        self.listening_thread = None

    def run(self):
        self.listening_thread = threading.Thread(target=self.start_listening)
        self.listening_thread.start()

    def start_listening(self):
        print("TCPServer waiting for connection ......")
        while True:
            self.server_socket.settimeout(5)
            if self.close_tag:
                break
            try:
                client_socket, client_address = self.server_socket.accept()
                print("TCPServer successfully connected by :%s" % str(client_address))
                self.socket_mapping[client_address] = client_socket
                self.connect_number += 1
            except socket.timeout:
                continue

    def receive_serializable_item_from(self, target_address):
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
        self.socket_mapping.get(target_address).close()
        self.socket_mapping.pop(target_address)
        self.connect_number -= 1

    def close_all(self):
        for address in self.socket_mapping:
            self.socket_mapping.get(address).close()
        self.socket_mapping.clear()
        self.connect_number = 0
        # shut down server
        self.close_tag = True
        self.listening_thread.join()
        self.server_socket.close()


class TCPClient(object):
    def __init__(self, self_address, self_port):
        self.target_address = None
        self.self_address = (self_address, self_port)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.client_socket.bind((self_address, self_port))

    def connect_to(self, host, port):
        self.target_address = (host, port)
        self.client_socket.connect(self.target_address)
        print("successfully connect to server %s:%d" % (host, port))

    def connect_to_with_retry(self, host, port):
        while True:
            try:
                self.connect_to(host, port)
                break
            except ConnectionRefusedError:
                print('failed to connect to server: %s:%d' % (host, port))
                time.sleep(1)
                continue

    def send_msg(self, msg):
        send_binary_data(self.client_socket, msg)

    def send_serializable_item(self, item):
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
        self.client_socket.close()


def send_binary_data(__socket, data):
    __socket.send(data)


def receive_binary_data(__socket, msg_size):
    return __socket.recv(msg_size)
