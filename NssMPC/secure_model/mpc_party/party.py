#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
For normal communication, Communicator is set to NCommunicator.
For multi-thread communication, Communicator is set to MCommunicator.
"""
import random
import time
from multiprocessing import Pipe, Lock

from NssMPC.common.random.prg import MT19937_PRG
from NssMPC.config import DEBUG_LEVEL, BIT_LEN, SOCKET_NUM, SOCKET_TYPE, DEVICE
from NssMPC.crypto.aux_parameter import MatmulTriples, RssMatmulTriples
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.vsigma_key import VSigmaKey
from NssMPC.secure_model.utils.buffer_thread.buffer_thread import BufferThread

if SOCKET_TYPE == 0:
    from NssMPC.common.network.communicator import NCommunicator as Communicator
else:
    from NssMPC.common.network.communicator import MCommunicator as Communicator


class PartyBase(object):
    """
    A class representing a participating party. Each Party object has a unique party_id to identify it.
    """

    def __init__(self, party_id):
        """
        ATTRIBUTES:
            * **communicator** (:class:`~NssMPC.common.network.communicator.NCommunicator`): participant
            * **party_id** (*int*): The unique integer ID that identifies each Party instance.
            * **providers** (*dict*): Parameter provider
            * **provider_threads** (*dict*): Thread provider

        """
        self.communicator = Communicator()
        self.party_id = party_id
        self.providers = {}
        self.provider_threads = {}

    def load_aux_params(self):
        """
        Load auxiliary parameters.

        Iterate through all providers of the current participant, loading the corresponding parameters according to the different parameter types:
            If it is MatmulTriples or MaliciousCMPKey or RssMatmulTriples, no processing is performed.

        then the :meth:`~NssMPC.secure_model.mpc_party.party.VirtualParty2PC.load_aux_params` method is recursively
        called before and after the virtual participant.If the provider's parameter type is not of some particular
        type, the corresponding parameter file is loaded, and the party attribute of the secondary parameter is set
        to the same as ``self``.

        """
        for name, provider in self.providers.items():
            if provider.param_type == MatmulTriples:
                continue
            if provider.param_type == VSigmaKey:
                continue
            if provider.param_type == RssMatmulTriples:
                continue
            param_saved_name = provider.saved_name + '_' + str(self.party_id) + '.pth'
            provider.load_param(saved_name=param_saved_name)

    def append_provider(self, provider):
        """
        This function is only used to add a provider to the Party, where the provider must be initialized and a name
        must be specified. If you are not currently debugging or are in a level 0 debug level, a buffer thread is
        created for each provider.

        .. note::
            append no longer uses class names as dictionary keys to avoid the problem of one party being unable to load two identical Parameter objects (in a three-party scenario where two parties execute 22-share computations together)

        :param provider: Parameter provider
        :type provider: ParamProvider
        """
        # append不再使用类名做字典的键值，避免一个party无法载入两个相同的Parameter的问题（3方两两执行22分享的计算场景）
        self.providers[provider.param_tag] = provider
        if not DEBUG_LEVEL:
            self.provider_threads[provider.param_tag] = BufferThread(provider, self, pipe=Pipe(True), lock=Lock())

    def get_param(self, param_tag, *args):
        """
        Get the parameter of a certain type.

        Based on the provided ``param_tag`` string or type, call the corresponding provider's ``get_parameters`` method and pass additional parameters.

        :param param_tag: A label used to identify parameters
        :type param_tag: str
        :param args: the arguments needed to get the parameter. The number of arguments is determined by the type of parameter.
        :type args: Any
        """

        if isinstance(param_tag, str):
            return self.providers[param_tag].get_parameters(*args)

        else:
            return self.providers[param_tag.__name__].get_parameters(*args)


class Party3PC(PartyBase):
    """
    An implementation specifically for 3-party computation. It achieves secure computation by establishing
    communication and coordination with the other two parties.
    """

    def __init__(self, party_id):
        """
        The constructor initializes the virtual participant and address mapping by calling the constructor of the parent class and initializing the virtual participant and address mapping.

        ATTRIBUTES:
            * **virtual_party_with_next** (:class:`VirtualParty2PC`): participant
            * **virtual_party_with_previous** (:class:`VirtualParty2PC`): Used to store parameters.
            * **prg_0** (:class:`~NssMPC.common.random.prg.MT19937_PRG`): Pseudo-random number generators
            * **prg_1** (:class:`~NssMPC.common.random.prg.MT19937_PRG`): Pseudo-random number generators
            * **sending_address** (*dict*): The address to send the data.
            * **receiving_address** (*dict*): The address to receive the data.
            * **target_address** (*dict*): The destination address of the communication. Used to determine the downstream and upstream connections.

        """
        super().__init__(party_id)
        self.virtual_party_with_next = VirtualParty2PC(1, self)
        self.virtual_party_with_previous = VirtualParty2PC(0, self)
        self.prg_0 = None
        self.prg_1 = None
        # 在Party这个抽象类里储存连接映射，Communicator只负责接收地址进行收发
        self.receiving_address_mapping = {}
        self.sending_address_mapping = {}
        self.target_address_mapping = {}

    def online(self):
        """
        This method is responsible for setting the online status of the participants.

        * First import the corresponding socket configuration *SOCKET_CONFIG* from the configuration file based on party_id.
        * Then set the send, receive, and destination address mappings for communication with the next and previous participants, and then set the maximum number of connections to 2. Call :meth:`set_communicator_address` to set the local communication address and port.
        * The server is then initialized and its listening address is set to connect to the address of the next and previous participant to establish two-way communication. Call the :meth:`check_all_parties_online` method to confirm that all participants have successfully come online.
        * Pseudo-random generator seeds are generated, auxiliary parameters are loaded, and finally all provider threads, including the threads of the current participant and the two virtual participants, are started to handle subsequent computation tasks

        """
        if self.party_id == 0:
            from NssMPC.config.configs import SOCKET_P0 as SOCKET_CONFIG
        elif self.party_id == 1:
            from NssMPC.config.configs import SOCKET_P1 as SOCKET_CONFIG
        else:
            from NssMPC.config.configs import SOCKET_P2 as SOCKET_CONFIG

        self.sending_address_mapping[(self.party_id + 1) % 3] = (
            SOCKET_CONFIG.TO_NEXT["ADDRESS"], SOCKET_CONFIG.TO_NEXT["PORT"])
        self.sending_address_mapping[(self.party_id - 1) % 3] = (
            SOCKET_CONFIG.TO_PREVIOUS["ADDRESS"], SOCKET_CONFIG.TO_PREVIOUS["PORT"])

        self.receiving_address_mapping[(self.party_id + 1) % 3] = (
            SOCKET_CONFIG.FROM_NEXT["ADDRESS"], SOCKET_CONFIG.FROM_NEXT["PORT"])
        self.receiving_address_mapping[(self.party_id - 1) % 3] = (
            SOCKET_CONFIG.FROM_PREVIOUS["ADDRESS"], SOCKET_CONFIG.FROM_PREVIOUS["PORT"])

        self.target_address_mapping[(self.party_id + 1) % 3] = (SOCKET_CONFIG.ADDRESS_NEXT, SOCKET_CONFIG.PORT_NEXT)
        self.target_address_mapping[(self.party_id - 1) % 3] = (
            SOCKET_CONFIG.ADDRESS_PREVIOUS, SOCKET_CONFIG.PORT_PREVIOUS)
        self.communicator.set_max_connections(2)
        self.set_communicator_address((SOCKET_CONFIG.ADDRESS, SOCKET_CONFIG.PORT))

        self.communicator.init_server()

        self.communicator.connect_to_other(other_address=self.target_address_mapping[(self.party_id + 1) % 3],
                                           socket_address=self.sending_address_mapping[(self.party_id + 1) % 3])
        self.communicator.connect_to_other(other_address=self.target_address_mapping[(self.party_id - 1) % 3],
                                           socket_address=self.sending_address_mapping[(self.party_id - 1) % 3])
        self.check_all_parties_online()
        print("Server", self.party_id, "is now online")
        self.generate_prg_seed()
        self.load_aux_params()
        for provider_thread in self.provider_threads.values():
            provider_thread.start()
        for provider_thread in self.virtual_party_with_previous.provider_threads.values():
            provider_thread.start()
        for provider_thread in self.virtual_party_with_next.provider_threads.values():
            provider_thread.start()

    def check_all_parties_online(self):
        """
        Check if all parties are online

        Before the check begins, the method pauses for 3 seconds. This can be to give the network connection and
        other participants some time to come online.

        The number of connections required is determined by the value of SOCKET_TYPE:
            * If *SOCKET_TYPE* is true, the connection number num will be **2** * *SOCKET_NUM*, which means bidirectional connections.
            * Otherwise, num is set to **2**, which means that only two other parties need to be connected.
        """
        time.sleep(3)
        num = 2 * SOCKET_NUM if SOCKET_TYPE else 2
        while self.communicator.connected_num() < num:
            # print(self.communicator.connected_num())
            pass

    def generate_prg_seed(self):
        """
        Generate two pseudorandom number generator (PRG) seeds.

        ``prg_seed_0`` is a random seed generated by the current party and sends ``prg_seed_0`` to the previous participant.
        It then receives the seed ``prg_seed_1`` from the next participant. Using the PRG of the Mersenne Twister
        algorithm, set the seeds ``prg_seed_0`` and ``prg_seed_1`` using the :meth:`~NssMPC.common.random.prg.MT19937_PRG.set_seeds` method.
        """
        prg_seed_0 = random.randint(0, 2 ** BIT_LEN)
        self.send((self.party_id - 1) % 3, prg_seed_0)
        prg_seed_1 = self.receive((self.party_id + 1) % 3)
        self.prg_0 = MT19937_PRG()
        self.prg_0.set_seeds(prg_seed_0)
        self.prg_1 = MT19937_PRG()
        self.prg_1.set_seeds(prg_seed_1)

    def set_communicator_address(self, address):
        """
        Set the address of the communication device.

        First, a parameter check is performed to ensure that the address passed in is a tuple. Then use the
        communicator property to call :meth:`~NssMPC.common.network.communicator.Communicator.set_address` to set the communication address and port

        :param address: Communication address of the local device
        :type address: tuple
        """
        assert isinstance(address, tuple)
        self.communicator.set_address(address[0], address[1])

    def send(self, target_id, x):
        """
        Send data **x** to the target address.

        Use the :meth:`~NssMPC.common.network.communicator.Communicator.send_to_address` method of ``self.communicator`` to send data to the address specified by ``target_address_mapping[target_id]``.

        :param target_id: The index value of the target address in the dictionary
        :type target_id: int
        :param x: Information that needs to be sent.
        :type x: Any

        """
        self.communicator.send_to_address(self.target_address_mapping[target_id], x)

    def receive(self, target_id):
        """
        Receive data from a specified address.

        Call the :meth:`~NssMPC.common.network.communicator.Communicator.recv_from_address` method of ``self.communicator`` to receive data from ``self.receiving_address_mapping[target_id]``.
        Check whether the received data is an instance of ArithmeticSecretSharing or ReplicatedSecretSharing. If it is, set the party attribute of the current object to it.

        :param target_id: The index value of the target address in the dictionary
        :type target_id: int
        :return: received data
        :rtype: ArithmeticSecretSharing or ReplicatedSecretSharing
        """
        ret = self.communicator.recv_from_address(self.receiving_address_mapping[target_id])
        from NssMPC.crypto.primitives.arithmetic_secret_sharing import ArithmeticSecretSharing
        from NssMPC.crypto.primitives.arithmetic_secret_sharing import ReplicatedSecretSharing
        if isinstance(ret, (ArithmeticSecretSharing, ReplicatedSecretSharing)):
            ret.party = self
        return ret

    def close(self):
        """
        Close the TCP connection and clean up the resources associated with it.

        The :meth:`~NssMPC.common.network.communicator.Communicator.close` method is called to close the TCP connection with the other
        participant.

        Then send signal **0** to the current process using the ``os.kill`` function.

        .. note::
            Sending the signal **0** does not actually kill the process, but is used to check if the process still exists. This is often used to verify the state of a process.

        """
        self.communicator.close()
        # for provider_thread in self.provider_threads.values():
        #     provider_thread.join()
        import os
        os.kill(os.getpid(), 0)


class VirtualParty2PC(PartyBase):
    """
    The implementation of virtual participants, which inherits from the Party class, completes relevant operations in multi-party computation through interactions with real participants.
    """

    def __init__(self, party_id, real_party):
        """
        :param party_id: The unique integer ID that identifies each Party instance.
        :type party_id: int
        :param real_party: A real participant object, providing the ability to communicate with other participants.
        :type real_party: PartyBase
        """
        super().__init__(party_id)
        self.real_party = real_party
        self.other_id = (real_party.party_id - party_id - 1) % 3

    def load_aux_params(self):
        """
        Load auxiliary parameters.

        Iterate through all providers of the current participant, loading the corresponding parameters according to the different parameter types:
            * if it is VOSKey or VSigmaKey, ``param_saved_name`` is set to **None**
            * if it is other data type, The save name of the build parameter ``param_saved_name`` is in the format of ``{provider.saved_name}_{kid}_{self.party_id}.pth``.
        The parameters are then loaded according to the parameter path. If the provider's parameter type is not of some particular
        type, the corresponding parameter file is loaded, and the party attribute of the secondary parameter is set
        to the same as ``self``.

        """
        for name, provider in self.providers.items():
            kid = (self.real_party.party_id + 1 + self.party_id) % 3
            param_saved_name = f'{provider.saved_name}_{kid}_{self.party_id}.pth'
            if name == 'VOSKey' or name == 'VSigmaKey':
                param_saved_name = None
            provider.load_param(saved_name=param_saved_name)

    def send(self, x):
        """
        Send messages to real participants.

        Call the real_party's send method to send data **x** to the participant specified by other_id.

        .. note::
            This means that the virtual participant does not actually perform the data transmission directly, but relies on the communication mechanism of the real participant.

        :param x: Information that needs to be sent.
        :type x: Any
        """
        self.real_party.send(self.other_id, x)

    def receive(self):
        """
        Receive messages from real participants.

        Call the receive method of the real_party object to receive a message from the other_id. If the returned data
        ret has a party attribute, set it to the current virtual participant.

        :return: Received data
        :rtype: Any
        """
        ret = self.real_party.receive(self.other_id)
        if hasattr(ret, 'party'):
            ret.party = self
        if hasattr(ret, 'to'):
            return ret.to(DEVICE)
        return ret
