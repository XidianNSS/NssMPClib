#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
For normal communication, Communicator is set to NCommunicator.
For multi-thread communication, Communicator is set to MCommunicator.
"""
import contextvars
import random
from multiprocessing import Pipe, Lock

from nssmpc.config import DEBUG_LEVEL, BIT_LEN, DEVICE
from nssmpc.infra.mpc.aux_parameter import Parameter, ParamProvider
from nssmpc.infra.mpc.communication import TensorPipeCommunicator
from nssmpc.infra.mpc.aux_parameter.buffer_thread import BufferThread
from nssmpc.infra.prg import MT19937_PRG

PartyCtx = contextvars.ContextVar('PartyCtx', default=None)


class Party:
    """
    A class representing a participating party with thread-safe runtime context management.
    Each Party object has a unique party_id to identify it.
    """

    def __init__(self, party_id, thread_model):
        """
        Initialize a Party instance.

        Args:
            party_id (int): The unique integer ID that identifies each Party instance.
            thread_model (dict): The threading model config to be used.

        Attributes:
            communicator (NCommunicator): participant communicator
            party_id (int): The unique integer ID that identifies each Party instance.
            thread_model (dict): The threading model config to be used
            providers (dict): Parameter providers
            provider_threads (dict): Thread providers

        Examples:
            party = Party(0)
        """
        self.communicator = None
        self.party_id = party_id
        self.thread_model_cfg = thread_model
        self.providers = {}
        self.provider_threads = {}

    def __repr__(self):
        return f"Party(party_id={self.party_id},thread_model={self.thread_model_cfg})"

    def online(self):
        """
        Initialize communication connections and verify the online status of all participants.

        Examples:
            party.online()
        """
        ...

    def send(self, *args):
        """
        Send data to other parties.

        Args:
            *args: Arguments for sending data.

        Examples:
            party.send(data)
        """
        ...

    def recv(self, *args):
        """
        Receive data from other parties.

        Args:
            *args: Arguments for receiving data.

        Returns:
            Any: Received data.

        Examples:
            data = party.recv()
        """
        ...

    def wait(self):
        """
        Ensure synchronisation of communications between the two parties.

        Examples:
            party.wait()
        """
        self.communicator.barrier()

    def close(self):
        """
        Close the TCP connection and clean up the resources associated with it.

        Examples:
            party.close()
        """
        self.communicator.shutdown()
        for provider_thread in self.provider_threads.values():
            provider_thread.join()

    def load_provider(self, param_type, tag):
        """Loads a parameter provider for the specified party.

        Args:
            param_type (Parameter): The type of parameter to be loaded.
            tag: The tag of the provider.

        Returns:
            ParamProvider: The loaded parameter provider.

        Examples:
            >>> provider = Parameter.load_provider(BeaverTriples)
        """
        provider = ParamProvider(param_type=param_type, saved_name=tag)

        try:
            provider.load_param_from_file(f'p{self.party_id}_{tag}.pth')
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Parameter type '{param_type}' with key '{tag}' not found for party {self.party_id}. "
                f"Please ensure the parameter file exists.")

        self.providers[tag] = provider
        if not DEBUG_LEVEL:
            self.provider_threads[tag] = BufferThread(provider, self, pipe=Pipe(True), lock=Lock())
            self.provider_threads[tag].start()
        return provider

    def get_param(self, param_type: Parameter, num: int, tag: str = None):
        """
        Get the parameter of a certain type.

        Based on the provided ``param_tag`` string or type, call the corresponding provider's
        ``get_parameters`` method and pass additional parameters.

        Args:
            param_type (Parameter): A label used to identify parameters.
            *args: The arguments needed to get the parameter.

        Returns:
            Any: The requested parameter.

        Examples:
            param = party.get_param(BeaverTriples, shape)
        """
        param_tag = param_type.__name__
        if tag is not None:
            param_tag += f'_{tag}'

        provider = self.providers.get(param_tag)
        if provider is None:
            self.load_provider(param_type, tag=param_tag)
            provider = self.providers.get(param_tag)

        return provider.get_parameters(num)


class Party2PC(Party):
    def __init__(self, party_id, threat_model, master_ip='localhost', master_port='29500'):
        """
        Constructor that initializes the class SemiHonestCS.

        Determine whether the current instance is a client or a server, and set party_id based on type.

        Args:
            type (str): Type of the current instance ('client' or 'server').
            master_ip (str): IP address of the master node. Defaults to 'localhost'.
            master_port (str): Port of the master node. Defaults to '29500'.

        Attributes:
            type (str): Type of the current instance.
            master_ip (str): IP address of the master node.
            master_port (str): Port of the master node.

        Examples:
            party = Party2PC('server')
        """
        assert party_id in [0, 1], "party_id must be 0 (server) or 1 (client)"
        self.party_id = party_id
        self.master_ip = master_ip
        self.master_port = master_port
        super(Party2PC, self).__init__(party_id, threat_model)

    def online(self):
        """
        Initialize communication connections and verify the online status of all participants.

        Examples:
            party.online()
        """
        self.communicator = TensorPipeCommunicator(self.party_id, 2, self.master_ip, self.master_port)
        self.communicator.wait_for_peers(30)
        for provider_thread in self.provider_threads.values():
            provider_thread.start()

    def send(self, x):
        """
        Send data to the target address.

        Args:
            x (Any): Information that needs to be sent.

        Examples:
            party.send(data)
        """
        self.communicator.send(self.party_id ^ 1, x)

    def recv(self):
        """
        Receive data from a specified address.

        Returns:
            ArithmeticSecretSharing or ReplicatedSecretSharing: Received data.

        Examples:
            data = party.recv()
        """
        return self.communicator.recv(self.party_id ^ 1)


class Party3PC(Party):
    """
    An implementation specifically for 3-party computation. It achieves secure computation by establishing
    communication and coordination with the other two parties.
    """

    def __init__(self, party_id: int, threat_model: dict, master_ip='localhost', master_port='29500'):
        """
        Initialize the 3PC party.

        Args:
            party_id (int): The unique integer ID.
            threat_model (dict): The threat model config for the party.
            master_ip (str): IP address of the master node. Defaults to 'localhost'.
            master_port (str): Port of the master node. Defaults to '29500'.

        Attributes:
            virtual_party_with_next (VirtualParty2PC): Virtual party for next neighbor.
            virtual_party_with_previous (VirtualParty2PC): Virtual party for previous neighbor.
            prg_0 (MT19937_PRG): Pseudo-random number generator 0.
            prg_1 (MT19937_PRG): Pseudo-random number generator 1.

        Examples:
            party = Party3PC(0)
        """
        super().__init__(party_id, threat_model)
        self.virtual_party_with_next = VirtualParty2PC(1, self)
        self.virtual_party_with_previous = VirtualParty2PC(0, self)
        self.prg_0 = None
        self.prg_1 = None
        # 在Party这个抽象类里储存连接映射，Communicator只负责接收地址进行收发
        self.master_ip = master_ip
        self.master_port = master_port

    def online(self):
        """
        Set the online status of the participants.

        Initializes communicator, waits for peers, generates PRG seeds, and starts provider threads.

        Examples:
            party.online()
        """
        self.communicator = TensorPipeCommunicator(self.party_id, 3, self.master_ip, self.master_port)
        self.communicator.wait_for_peers(30)
        print("Server", self.party_id, "is now online")
        self.generate_prg_seed()
        for provider_thread in self.provider_threads.values():
            provider_thread.start()
        for provider_thread in self.virtual_party_with_previous.provider_threads.values():
            provider_thread.start()
        for provider_thread in self.virtual_party_with_next.provider_threads.values():
            provider_thread.start()

    def generate_prg_seed(self):
        """
        Generate two pseudorandom number generator (PRG) seeds.

        Exchanges seeds with neighbors and initializes PRGs.

        Examples:
            party.generate_prg_seed()
        """
        prg_seed_0 = random.randint(0, 2 ** BIT_LEN)
        self.send((self.party_id - 1) % 3, prg_seed_0)
        prg_seed_1 = self.recv((self.party_id + 1) % 3)
        self.prg_0 = MT19937_PRG()
        self.prg_0.set_seeds(prg_seed_0)
        self.prg_1 = MT19937_PRG()
        self.prg_1.set_seeds(prg_seed_1)

    def send(self, target_id, x):
        """
        Send data to the target address.

        Args:
            target_id (int): The index value of the target address.
            x (Any): Information that needs to be sent.

        Examples:
            party.send(1, data)
        """
        self.communicator.send(target_id, x)

    def recv(self, target_id):
        """
        Receive data from a specified address.

        Args:
            target_id (int): The index value of the target address.

        Returns:
            Any: Received data.

        Examples:
            data = party.recv(1)
        """
        ret = self.communicator.recv(target_id)
        return ret

    def close(self):
        """
        Close the TCP connection and clean up resources.

        Examples:
            party.close()
        """
        self.communicator.shutdown()
        # for provider_thread in self.provider_threads.values():
        #     provider_thread.join()


class VirtualParty2PC(Party):
    """
    The implementation of virtual participants, which inherits from the Party class, completes relevant operations in multi-party computation through interactions with real participants.
    """

    def __init__(self, party_id, real_party):
        """
        Initialize a VirtualParty2PC instance.

        Args:
            party_id (int): The unique integer ID that identifies each Party instance.
            real_party (Party): A real participant object.

        Examples:
            v_party = VirtualParty2PC(0, real_party)
        """
        super().__init__(party_id, real_party.thread_model_cfg)
        self.real_party = real_party
        self.other_id = (real_party.party_id - party_id - 1) % 3

    def send(self, x):
        """
        Send messages to real participants.

        Args:
            x (Any): Information that needs to be sent.

        Examples:
            v_party.send(data)
        """
        self.real_party.send(self.other_id, x)

    def recv(self):
        """
        Receive messages from real participants.

        Returns:
            Any: Received data.

        Examples:
            data = v_party.receive()
        """
        ret = self.real_party.recv(self.other_id)
        if hasattr(ret, 'to'):
            return ret.to(DEVICE)
        return ret
