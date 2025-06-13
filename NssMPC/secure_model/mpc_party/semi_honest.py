#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
In the semi-honest model, the parties (including third-party entities) follow the protocol's instructions for
computation, but may attempt to infer the private data of other parties from the information they obtain during the
protocol execution.
"""
from NssMPC.common.network.async_tcp import *
from NssMPC.config.configs import GE_TYPE
from NssMPC.crypto.aux_parameter import BooleanTriples, DICFKey, GrottoDICFKey, SigmaDICFKey, \
    AssMulTriples, ReciprocalSqrtKey, Wrap, DivKey, B2AKey, TanhKey
from NssMPC.crypto.aux_parameter.truncation_keys.rss_trunc_aux_param import RssTruncAuxParams
from NssMPC.secure_model.mpc_party.party import PartyBase, Party3PC
from NssMPC.secure_model.utils.param_provider import MatrixBeaverProvider
from NssMPC.secure_model.utils.param_provider import ParamProvider


class SemiHonestCS(PartyBase):
    """
    This class supports two types: client or server. Each type of entity has its specific party_id: 1 for clients and 0 for servers.
    Also, each entity has TCP connection parameters associated with it, as well as default data types and scaling.
    """

    def __init__(self, type='client'):
        """
        Constructor that initializes the class SemiHonestCS.

        Determine whether the current instance is a client or a server, and set party_id based on type. Then define the properties to store the send, receive, and destination addresses.

        ATTRIBUTES:
            * **type** (*Type*): Type of the current instance.
            * **sending_address** (*tuple*): The address to send the data.
            * **receiving_address** (*tuple*): The address to receive the data.
            * **target_address** (*tuple*): The destination address of the communication. Used to determine the downstream and upstream connections.

        .. note::
            The connection between **SemiHonestCS** and **Communicator** is:
                The Party's abstract class SemiHonestCS stores connection mappings, and Communicator is only responsible for receiving addresses for sending and receiving.
        """
        assert type in ('client', 'server'), "type must be 'client' or 'server'"
        self.type = type
        party_id = 0 if type == 'server' else 1
        super(SemiHonestCS, self).__init__(party_id)

        # 在Party这个抽象类里储存连接映射，Communicator只负责接收地址进行收发
        self.sending_address = None
        self.receiving_address = None
        self.target_address = None

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

    def online(self):
        """
        Initialize network connections and verify the online status of all participants.

        The party_id determines which socket configuration is used to set up ``sending_address``,
        ``receiving_address``, and ``target_address``.
        Then :meth:`~NssMPC.common.network.communicator.Communicator.set_max_connections` is used to set the maximum
        number of connections to 1, :meth:`set_communicator_address` is used to set the target communication address and
        port, :meth:`~NssMPC.common.network.communicator.NCommunicator.init_server()` is used to initialize the server, and :meth:`~NssMPC.common.network.communicator.NCommunicator.connect_to_other` starts connecting to other entities.
        Then call :meth:`check_all_parties_online` to verify that all participants are online. The auxiliary parameters are loaded by calling :meth:`~NssMPC.secure_model.mpc_party.party.Party.load_aux_params`.
        Finally, start the load helper parameter thread.

        """
        if self.party_id == 0:
            from NssMPC.config.configs import SOCKET_P0 as SOCKET_CONFIG
            self.sending_address = (SOCKET_CONFIG.TO_NEXT["ADDRESS"], SOCKET_CONFIG.TO_NEXT["PORT"])
            self.receiving_address = (SOCKET_CONFIG.FROM_NEXT["ADDRESS"], SOCKET_CONFIG.FROM_NEXT["PORT"])
            self.target_address = (SOCKET_CONFIG.ADDRESS_NEXT, SOCKET_CONFIG.PORT_NEXT)
        else:
            from NssMPC.config.configs import SOCKET_P1 as SOCKET_CONFIG
            self.sending_address = (SOCKET_CONFIG.TO_PREVIOUS["ADDRESS"], SOCKET_CONFIG.TO_PREVIOUS["PORT"])
            self.receiving_address = (SOCKET_CONFIG.FROM_PREVIOUS["ADDRESS"], SOCKET_CONFIG.FROM_PREVIOUS["PORT"])
            self.target_address = (SOCKET_CONFIG.ADDRESS_PREVIOUS, SOCKET_CONFIG.PORT_PREVIOUS)
        self.communicator.set_max_connections(1)
        self.set_communicator_address((SOCKET_CONFIG.ADDRESS, SOCKET_CONFIG.PORT))
        self.communicator.init_server()
        self.communicator.connect_to_other(other_address=self.target_address,
                                           socket_address=self.sending_address)

        # self.communicator.connect()
        self.check_all_parties_online()
        self.load_aux_params()
        # 加载辅助参数

        for provider_thread in self.provider_threads.values():
            provider_thread.start()

    def check_all_parties_online(self):
        """
        Check if all parties are online

        Before the check begins, the method pauses for 3 seconds. This can be to give the network connection and
        other participants some time to come online. A loop is then entered until at least one participant is
        connected (i.e., connected_num() returns a value greater than or equal to 1).

        """
        time.sleep(3)
        while self.communicator.connected_num() < 1:
            pass

    def set_comparison_provider(self):
        """
        Provide parameters related to comparison operations.

        CMPMapping is a mapping dictionary that maps a string to a corresponding class. Then, the parameter type
        corresponding to *GE_TYPE* obtained from CMPMapping is passed to the ParamProvider as a parameter type, and the result is
        added to the list as a new parameter provider. Analogously, add GrottoDICFKey for equality operations. Add B2AKey for boolean to arithmetic conversion.

        """
        CMPMapping = {
            'MSB': BooleanTriples,
            'DICF': DICFKey,
            'PPQ': GrottoDICFKey,
            'SIGMA': SigmaDICFKey
        }
        self.append_provider(ParamProvider(param_type=CMPMapping[GE_TYPE]))
        self.append_provider(ParamProvider(param_type=GrottoDICFKey))  # EQUAL
        self.append_provider(ParamProvider(param_type=B2AKey))  # B2A

    def set_multiplication_provider(self):
        """
        Provide parameters related to multiplication operations.

        Add AssMulTriples for Arithmetic secrets shared matrix multiplication operations.
        Add MatrixBeaverProvider for creating matrix Beaver Triples.
        Add Wrap as a parameter provider for packaging.
        """
        self.append_provider(ParamProvider(param_type=AssMulTriples, saved_name='2PCBeaver'))
        self.append_provider(MatrixBeaverProvider(party=self))
        self.append_provider(ParamProvider(param_type=Wrap))

    def set_nonlinear_operation_provider(self):
        """
        Add DivKey for division operations.
        Add ReciprocalSqrtKey for finding the reciprocal of the square root.
        Add B2AKey for Boolean to arithmetic conversion.
        Add TanhKey for hyperbolic tangent calculation.
        """
        self.append_provider(ParamProvider(param_type=DivKey))
        self.append_provider(ParamProvider(param_type=ReciprocalSqrtKey))
        self.append_provider(ParamProvider(param_type=B2AKey))
        self.append_provider(ParamProvider(param_type=TanhKey))

    def send(self, x):
        """
        Send data **x** to the target address.

        Use the :meth:`~NssMPC.common.network.communicator.Communicator.send_to_address` method of ``self.communicator`` to send data to the address specified by ``self_target_address``.

        :param x: Information that needs to be sent.
        :type x: Any
        """
        self.communicator.send_to_address(self.target_address, x)

    def receive(self):
        """
        Receive data from a specified address.

        Call the :meth:`~NssMPC.common.network.communicator.Communicator.recv_from_address` method of ``self.communicator`` to receive data from ``self.receiving_address``.
        If the received data object ``ret`` has a ``party`` attribute, it is set to the current object.

        :return: Received data
        :rtype: Any
        """
        ret = self.communicator.recv_from_address(self.receiving_address)
        return ret

    def wait(self):
        """
        Ensure synchronisation of communications between the two parties.

        First send a value of **0** to indicate that you are ready or waiting for the other party.
        Then call the :meth:`receive` method and wait to receive the response from the other party.
        """
        self.send(0)
        self.receive()

    def close(self):
        """
        Close the TCP connection and clean up the resources associated with it.

        First, the :meth:`~NssMPC.common.network.communicator.Communicator.close` method is called to close the TCP connection with the other
        participant, then all the threads in the ``self.provider_threads`` dictionary are traversed, and the ``join()``
        method is called.

        .. note::
            The ``join()`` method causes the main thread to wait for each provider thread to complete its execution. This is to ensure that all relevant calculations or operations have been completed before closing the connection.

        Then send signal **0** to the current process using the ``os.kill`` function.

        .. note::
            Sending the signal **0** does not actually kill the process, but is used to check if the process still exists. This is often used to verify the state of a process.
        """
        self.communicator.close()
        for provider_thread in self.provider_threads.values():
            provider_thread.join()
        import os
        os.kill(os.getpid(), 0)


class SemiHonest3PCParty(Party3PC):
    """
    Participants in 3-party computation. This class contains multiple functionalities, allowing participants to
    communicate with each other, send and receive data, and load and use keys and masks.
    """

    def __init__(self, id):
        """
        Initializes an instance of PartyMPC.

        If the ID is not 2, then the set_comparison_provider method is called to set the comparison provider.

        :param id: This participant's ID.
        :type id: int
        """
        super(SemiHonest3PCParty, self).__init__(id)
        if id != 2:
            self.set_comparison_provider()

    def set_comparison_provider(self):
        """
        This method is used to set the comparison parameter provider for participants whose ID are not 2.

        By calling the :meth:`~NssMPC.secure_model.mpc_party.party.Party.append_provider` method, the SigmaDICFKey provider is added to the provider list as a parameter provider.
        """
        # pass
        if self.party_id != 2:
            self.append_provider(ParamProvider(SigmaDICFKey))
        # self.append_provider(GrottoDICFKey)  # EQUAL

    def set_trunc_provider(self):
        """
        This method is responsible for setting up the RSS truncation parameter provider.

        By calling the :meth:`~NssMPC.secure_model.mpc_party.party.Party.append_provider` method, the RssTruncAuxParams provider is added to the provider list as a parameter provider.
        """
        self.append_provider(ParamProvider(param_type=RssTruncAuxParams))

    def set_providers(self):
        """
        This method calls upon the above methods(:meth:`set_trunc_provider` and :meth:`~NssMPC.secure_model.mpc_party.party.Party.load_aux_params`) to complete
        the setup of all providers. Then Start all provider threads in ``provider_thread`` dictionary.

        """
        self.set_trunc_provider()
        self.load_aux_params()
        for provider_thread in self.provider_threads.values():
            provider_thread.start()
