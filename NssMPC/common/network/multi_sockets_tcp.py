import socket
import struct
import threading
import time

from NssMPC.config import SOCKET_MAX_SIZE, SOCKET_NUM


class TCPServer(object):
    def __init__(self, address, port):
        self.address = (address, port)
        self.server_socket = []
        self.socket_mapping = []
        self.connect_number = 0
        self.close_tag = False
        self.listening_thread = None
        self.socket_num = SOCKET_NUM
        self.start()

    def start(self):
        address, port = self.address
        for i in range(self.socket_num):
            self.server_socket.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            self.server_socket[i].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket[i].bind((address, port + i))
            self.server_socket[i].listen(5)
            self.socket_mapping.append({})

    def run(self):
        for i in range(self.socket_num):
            self.socket_mapping.append({})
            self.listening_thread = threading.Thread(target=self.start_listening, args=(i,))
            self.listening_thread.start()

    def start_listening(self, server_idx):
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
    def __init__(self, self_address, self_port):
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
        address, port = self.self_address
        for i in range(self.socket_num):
            self.client_socket.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            self.client_socket[i].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.client_socket[i].setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.client_socket[i].bind((address, port + i))

    def connect_to(self, host, port, idx):
        self.target_address.append((host, port + idx))
        return self.client_socket[idx].connect_ex(self.target_address[idx])

    def connect_to_with_retry(self, host, port, idx):
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
    __socket.send(data)


def receive_binary_data(__socket, msg_size):
    return __socket.recv(msg_size)


def receive_data(__socket, data_size):
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
    send_length = 0
    while send_length < len(msg):
        if len(msg) - send_length < SOCKET_MAX_SIZE:
            send_size = __socket.send(msg[send_length:])
        else:
            send_size = __socket.send(msg[send_length: send_length + SOCKET_MAX_SIZE])
        send_length += send_size
