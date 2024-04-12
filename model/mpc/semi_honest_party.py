from common.aux_parameter.buffer_thread import BufferThread
from common.aux_parameter.param_provider import ParamProvider
from common.network.async_tcp import *
from common.utils import count_bytes, bytes_convert
from config.base_configs import GE_TYPE, DEBUG_LEVEL
from crypto.primitives.arithmetic_secret_sharing.arithmetic_shared_ring_tensor import ArithmeticSharedRingTensor
from crypto.primitives.beaver.beaver_triples import BeaverTriples
from crypto.primitives.beaver.matrix_triples import MatrixTriples, MatrixBeaverProvider
from crypto.primitives.beaver.msb_triples import MSBTriples
from crypto.protocols.arithmetic_secret_sharing.truncate import Wrap
from crypto.protocols.function_secret_sharing import *


class Party(object):
    """
    A class representing a participating party.
    Each Party object has a unique party_id to identify it.

    Attributes:
        party_id (int): unique identifier representing the party
    """

    def __init__(self, party_id):
        self.party_id = party_id


class SemiHonestCS(Party):
    """
    Client-server model in a semi-honest setup.
    This class supports two types: client or server.
    Each type of entity has its specific party_id: 1 for clients and 0 for servers.
    Also, each entity has TCP connection parameters associated with it, as well as default data types and scaling.
    """

    def __init__(self, type='client'):
        assert type in ('client', 'server'), "type must be 'client' or 'server'"
        self.type = type
        party_id = 0 if type == 'server' else 1
        super(SemiHonestCS, self).__init__(party_id)
        self.server = None
        self.client = None
        self.other_client_address = None
        self.providers = {}
        self.provider_threads = {}

        self.comm_rounds = {'send': 0, 'recv': 0}
        self.comm_bytes = {'send': 0, 'recv': 0}

    def set_server(self, address):
        assert isinstance(address, tuple)
        self.server = TCPServer(address[0], address[1])

    def set_client(self, address):
        assert isinstance(address, tuple)
        self.client = TCPClient(address[0], address[1])

    def set_client_address(self, address):
        self.other_client_address = address

    def connect(self, self_server_socket, self_client_socket, other_server_socket, other_client_socket):
        assert isinstance(self_server_socket, tuple)
        assert isinstance(self_client_socket, tuple)
        assert isinstance(other_server_socket, tuple)
        assert isinstance(other_client_socket, tuple)

        self.set_server(self_server_socket)
        self.set_client(self_client_socket)
        self.server.run()
        self.client.connect_to_with_retry(other_server_socket[0], other_server_socket[1])
        self.check_all_parties_online()
        self.set_client_address(other_client_socket)
        for provider_thread in self.provider_threads.values():
            provider_thread.start()

    def check_all_parties_online(self):
        """
        Check if all parties are online
        """
        while self.server.connect_number < 1:
            pass

    def append_provider(self, param_type):
        """
        Append a provider to the party.
        """
        if param_type not in self.providers:
            if param_type == MatrixTriples:
                self.providers[param_type] = MatrixBeaverProvider(self)
            else:
                self.providers[param_type] = ParamProvider(param_type, self)
                self.providers[param_type].load_param()
                if not DEBUG_LEVEL:
                    self.provider_threads[param_type] = BufferThread(self.providers[param_type], self)

    def get_param(self, param_type, *args):
        """
        Get the parameter of a certain type.
        *args: the arguments needed to get the parameter. The number of arguments is determined by the type of parameter.
        """
        return self.providers[param_type].get_parameters(*args)

    def set_comparison_provider(self):
        CMPMapping = {
            'MSB': MSBTriples,
            'DICF': DICFKey,
            'PPQ': PPQCompareKey,
            'SIGMA': SigmaCompareKey
        }
        self.append_provider(CMPMapping[GE_TYPE])
        self.append_provider(PPQCompareKey)

    def set_multiplication_provider(self):
        self.append_provider(BeaverTriples)
        self.append_provider(MatrixTriples)
        self.append_provider(Wrap)

    def send(self, x):
        self.comm_rounds['send'] += 1
        self.comm_bytes['send'] += count_bytes(x)
        self.client.send_serializable_item(x)

    def receive(self):
        ret = self.server.receive_serializable_item_from(self.other_client_address)
        self.comm_rounds['recv'] += 1
        self.comm_bytes['recv'] += count_bytes(ret)
        if isinstance(ret, ArithmeticSharedRingTensor):
            ret.party = self
        return ret

    def wait(self):
        """
        Ensure synchronisation of communications between the two parties
        """
        self.send(0)
        self.receive()

    def close(self):
        """
        Close the connection of tcp
        """
        self.client.close()
        self.server.close_all()
        for provider_thread in self.provider_threads.values():
            provider_thread.join()
        import os
        os.kill(os.getpid(), 0)
        print(f"Communication costs:\n\tsend rounds: {self.comm_rounds['send']}\t\t"
              f"send bytes: {bytes_convert(self.comm_bytes['send'])}.")
        print(f"\trecv rounds: {self.comm_rounds['recv']}\t\t"
              f"recv bytes: {bytes_convert(self.comm_bytes['recv'])}.")
