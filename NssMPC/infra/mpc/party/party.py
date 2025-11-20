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

from NssMPC.config import DEBUG_LEVEL, BIT_LEN, DEVICE, SOCKET_TYPE
from NssMPC.infra.mpc.communication import TensorPipeCommunicator
from NssMPC.infra.mpc.param_provider.buffer_thread import BufferThread
from NssMPC.infra.prg import MT19937_PRG

PartyCtx = contextvars.ContextVar('PartyCtx', default=None)

if SOCKET_TYPE == 0:
    from NssMPC.infra.mpc.communication import TensorPipeCommunicator as Communicator
else:
    from NssMPC.infra.mpc.communication import MCommunicator as Communicator


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

    def append_provider(self, provider):
        """
        Add a provider to the Party.

        The provider must be initialized and a name must be specified. If not currently debugging
        or in level 0 debug level, a buffer thread is created for each provider.

        Args:
            provider (ParamProvider): Parameter provider.

        Examples:
            party.append_provider(provider)
        """
        self.providers[provider.param_tag] = provider
        if not DEBUG_LEVEL:
            self.provider_threads[provider.param_tag] = BufferThread(provider, self, pipe=Pipe(True), lock=Lock())
            self.provider_threads[provider.param_tag].start()

    def get_param(self, param_type, *args, **kwargs):
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
        if kwargs.get('tag') is not None:
            provider = self.providers.get(kwargs['tag'])
            if provider is None:
                param_type.load_provider(self)
                provider = self.providers.get(kwargs['tag'])
            return provider.get_parameters(*args)

        provider = self.providers.get(param_type.__name__)
        if provider is None:
            param_type.load_provider(self)
            provider = self.providers.get(param_type.__name__)

        return provider.get_parameters(*args)


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


# class Party3PC(PartyBase):
#     """
#     An implementation specifically for 3-party computation. It achieves secure computation by establishing
#     communication and coordination with the other two parties.
#     """
#
#     def __init__(self, party_id):
#         """
#         The constructor initializes the virtual participant and address mapping by calling the constructor of the parent class and initializing the virtual participant and address mapping.
#
#         ATTRIBUTES:
#             * **virtual_party_with_next** (:class:`VirtualParty2PC`): participant
#             * **virtual_party_with_previous** (:class:`VirtualParty2PC`): Used to store parameters.
#             * **prg_0** (:class:`~NssMPC.infra.random.prg.MT19937_PRG`): Pseudo-random number generators
#             * **prg_1** (:class:`~NssMPC.infra.random.prg.MT19937_PRG`): Pseudo-random number generators
#             * **sending_address** (*dict*): The address to send the data.
#             * **receiving_address** (*dict*): The address to receive the data.
#             * **target_address** (*dict*): The destination address of the communication. Used to determine the downstream and upstream connections.
#
#         """
#         super().__init__(party_id)
#         self.virtual_party_with_next = VirtualParty2PC(1, self)
#         self.virtual_party_with_previous = VirtualParty2PC(0, self)
#         self.prg_0 = None
#         self.prg_1 = None
#         # 在Party这个抽象类里储存连接映射，Communicator只负责接收地址进行收发
#         self.receiving_address_mapping = {}
#         self.sending_address_mapping = {}
#         self.target_address_mapping = {}
#
#     def online(self):
#         """
#         This method is responsible for setting the online status of the participants.
#
#         * First import the corresponding socket configuration *SOCKET_CONFIG* from the configuration file based on party_id.
#         * Then set the send, receive, and destination address mappings for communication with the next and previous participants, and then set the maximum number of connections to 2. Call :meth:`set_communicator_address` to set the local communication address and port.
#         * The server is then initialized and its listening address is set to connect to the address of the next and previous participant to establish two-way communication. Call the :meth:`check_all_parties_online` method to confirm that all participants have successfully come online.
#         * Pseudo-random generator seeds are generated, auxiliary parameters are loaded, and finally all provider threads, including the threads of the current participant and the two virtual participants, are started to handle subsequent computation tasks
#
#         """
#         if self.party_id == 0:
#             from NssMPC.config.configs import SOCKET_P0 as SOCKET_CONFIG
#         elif self.party_id == 1:
#             from NssMPC.config.configs import SOCKET_P1 as SOCKET_CONFIG
#         else:
#             from NssMPC.config.configs import SOCKET_P2 as SOCKET_CONFIG
#
#         self.sending_address_mapping[(self.party_id + 1) % 3] = (
#             SOCKET_CONFIG.TO_NEXT["ADDRESS"], SOCKET_CONFIG.TO_NEXT["PORT"])
#         self.sending_address_mapping[(self.party_id - 1) % 3] = (
#             SOCKET_CONFIG.TO_PREVIOUS["ADDRESS"], SOCKET_CONFIG.TO_PREVIOUS["PORT"])
#
#         self.receiving_address_mapping[(self.party_id + 1) % 3] = (
#             SOCKET_CONFIG.FROM_NEXT["ADDRESS"], SOCKET_CONFIG.FROM_NEXT["PORT"])
#         self.receiving_address_mapping[(self.party_id - 1) % 3] = (
#             SOCKET_CONFIG.FROM_PREVIOUS["ADDRESS"], SOCKET_CONFIG.FROM_PREVIOUS["PORT"])
#
#         self.target_address_mapping[(self.party_id + 1) % 3] = (SOCKET_CONFIG.ADDRESS_NEXT, SOCKET_CONFIG.PORT_NEXT)
#         self.target_address_mapping[(self.party_id - 1) % 3] = (
#             SOCKET_CONFIG.ADDRESS_PREVIOUS, SOCKET_CONFIG.PORT_PREVIOUS)
#         self.communicator.set_max_connections(2)
#         self.set_communicator_address((SOCKET_CONFIG.ADDRESS, SOCKET_CONFIG.PORT))
#
#         self.communicator.init_server()
#
#         self.communicator.connect_to_other(other_address=self.target_address_mapping[(self.party_id + 1) % 3],
#                                            socket_address=self.sending_address_mapping[(self.party_id + 1) % 3])
#         self.communicator.connect_to_other(other_address=self.target_address_mapping[(self.party_id - 1) % 3],
#                                            socket_address=self.sending_address_mapping[(self.party_id - 1) % 3])
#         self.check_all_parties_online()
#         print("Server", self.party_id, "is now online")
#         self.generate_prg_seed()
#         self.load_aux_params()
#         for provider_thread in self.provider_threads.values():
#             provider_thread.start()
#         for provider_thread in self.virtual_party_with_previous.provider_threads.values():
#             provider_thread.start()
#         for provider_thread in self.virtual_party_with_next.provider_threads.values():
#             provider_thread.start()
#
#     def check_all_parties_online(self):
#         """
#         Check if all parties are online
#
#         Before the check begins, the method pauses for 3 seconds. This can be to give the communication connection and
#         other participants some time to come online.
#
#         The number of connections required is determined by the value of SOCKET_TYPE:
#             * If *SOCKET_TYPE* is true, the connection number num will be **2** * *SOCKET_NUM*, which means bidirectional connections.
#             * Otherwise, num is set to **2**, which means that only two other parties need to be connected.
#         """
#         time.sleep(3)
#         num = 2 * SOCKET_NUM if SOCKET_TYPE else 2
#         while self.communicator.connected_num() < num:
#             # print(self.communicator.connected_num())
#             pass
#
#     def generate_prg_seed(self):
#         """
#         Generate two pseudorandom number generator (PRG) seeds.
#
#         ``prg_seed_0`` is a random seed generated by the current party and sends ``prg_seed_0`` to the previous participant.
#         It then receives the seed ``prg_seed_1`` from the next participant. Using the PRG of the Mersenne Twister
#         algorithm, set the seeds ``prg_seed_0`` and ``prg_seed_1`` using the :meth:`~NssMPC.infra.random.prg.MT19937_PRG.set_seeds` method.
#         """
#         prg_seed_0 = random.randint(0, 2 ** BIT_LEN)
#         self.send((self.party_id - 1) % 3, prg_seed_0)
#         prg_seed_1 = self.receive((self.party_id + 1) % 3)
#         self.prg_0 = MT19937_PRG()
#         self.prg_0.set_seeds(prg_seed_0)
#         self.prg_1 = MT19937_PRG()
#         self.prg_1.set_seeds(prg_seed_1)
#
#     def set_communicator_address(self, address):
#         """
#         Set the address of the communication device.
#
#         First, a parameter check is performed to ensure that the address passed in is a tuple. Then use the
#         communicator property to call :meth:`~NssMPC.infra.communication.communicator.Communicator.set_address` to set the communication address and port
#
#         :param address: Communication address of the local device
#         :type address: tuple
#         """
#         assert isinstance(address, tuple)
#         self.communicator.set_address(address[0], address[1])
#
#     def send(self, target_id, x):
#         """
#         Send data **x** to the target address.
#
#         Use the :meth:`~NssMPC.infra.communication.communicator.Communicator.send_to_address` method of ``self.communicator`` to send data to the address specified by ``target_address_mapping[target_id]``.
#
#         :param target_id: The index value of the target address in the dictionary
#         :type target_id: int
#         :param x: Information that needs to be sent.
#         :type x: Any
#
#         """
#         self.communicator.send_to_address(self.target_address_mapping[target_id], x)
#
#     def receive(self, target_id):
#         """
#         Receive data from a specified address.
#
#         Call the :meth:`~NssMPC.infra.communication.communicator.Communicator.recv_from_address` method of ``self.communicator`` to receive data from ``self.receiving_address_mapping[target_id]``.
#         Check whether the received data is an instance of ArithmeticSecretSharing or ReplicatedSecretSharing. If it is, set the party attribute of the current object to it.
#
#         :param target_id: The index value of the target address in the dictionary
#         :type target_id: int
#         :return: received data
#         :rtype: ArithmeticSecretSharing or ReplicatedSecretSharing
#         """
#         ret = self.communicator.recv_from_address(self.receiving_address_mapping[target_id])
#         from NssMPC.crypto.primitives.secret_sharing import ArithmeticSecretSharing
#         from NssMPC.crypto.primitives.secret_sharing import ReplicatedSecretSharing
#         if isinstance(ret, (ArithmeticSecretSharing, ReplicatedSecretSharing)):
#             ret.party = self
#         return ret
#
#     def close(self):
#         """
#         Close the TCP connection and clean up the resources associated with it.
#
#         The :meth:`~NssMPC.infra.communication.communicator.Communicator.close` method is called to close the TCP connection with the other
#         participant.
#
#         Then send signal **0** to the current process using the ``os.kill`` function.
#
#         .. note::
#             Sending the signal **0** does not actually kill the process, but is used to check if the process still exists. This is often used to verify the state of a process.
#
#         """
#         self.communicator.close()
#         # for provider_thread in self.provider_threads.values():
#         #     provider_thread.join()
#         import os
#         os.kill(os.getpid(), 0)

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

    # def load_aux_params(self):
    #     """
    #     Load auxiliary parameters.
    #
    #     Iterate through all providers of the current participant, loading the corresponding parameters according to the different parameter types:
    #         * if it is VOSKey or VSigmaKey, ``param_saved_name`` is set to **None**
    #         * if it is other data type, The save name of the build parameter ``param_saved_name`` is in the format of ``{provider.saved_name}_{kid}_{self.party_id}.pth``.
    #     The parameters are then loaded according to the parameter path. If the provider's parameter type is not of some particular
    #     type, the corresponding parameter file is loaded, and the party attribute of the secondary parameter is set
    #     to the same as ``self``.
    #
    #     """
    #     for name, provider in self.providers.items():
    #         kid = (self.real_party.party_id + 1 + self.party_id) % 3
    #         param_saved_name = f'{provider.saved_name}_{kid}_{self.party_id}.pth'
    #         if name == 'VOSKey' or name == 'VSigmaKey':
    #             param_saved_name = None
    #         provider.load_param(saved_name=param_saved_name)

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
