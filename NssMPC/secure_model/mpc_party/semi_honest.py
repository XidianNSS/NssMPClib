from NssMPC.common.network.async_tcp import *
from NssMPC.config.configs import GE_TYPE
from NssMPC.crypto.aux_parameter import BooleanTriples, DICFKey, GrottoDICFKey, SigmaDICFKey, \
    AssMulTriples, ReciprocalSqrtKey, Wrap, DivKey, B2AKey, TanhKey
from NssMPC.crypto.aux_parameter.truncation_keys.rss_trunc_aux_param import RssTruncAuxParams
from NssMPC.secure_model.mpc_party.party import Party, Party3PC
from NssMPC.secure_model.utils.param_provider import MatrixBeaverProvider
from NssMPC.secure_model.utils.param_provider.param_provider import ParamProvider


class SemiHonestCS(Party):
    """
    Client-server secure_model in a semi-honest setup.
    This class supports two types: client or server.
    Each type of entity has its specific party_id: 1 for clients and 0 for servers.
    Also, each entity has TCP connection parameters associated with it, as well as default data types and scaling.
    """

    def __init__(self, type='client'):
        assert type in ('client', 'server'), "type must be 'client' or 'server'"
        self.type = type
        party_id = 0 if type == 'server' else 1
        super(SemiHonestCS, self).__init__(party_id)

        # 在Party这个抽象类里储存连接映射，Communicator只负责接收地址进行收发
        self.sending_address = None
        self.receiving_address = None
        self.target_address = None

    def set_communicator_address(self, address):
        assert isinstance(address, tuple)
        self.communicator.set_address(address[0], address[1])

    def online(self):
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
        """
        time.sleep(3)
        while self.communicator.connected_num() < 1:
            pass

    def set_comparison_provider(self):
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
        self.append_provider(ParamProvider(param_type=AssMulTriples, saved_name='2PCBeaver'))
        self.append_provider(MatrixBeaverProvider(party=self))
        self.append_provider(ParamProvider(param_type=Wrap))

    def set_nonlinear_operation_provider(self):
        self.append_provider(ParamProvider(param_type=DivKey))
        self.append_provider(ParamProvider(param_type=ReciprocalSqrtKey))
        self.append_provider(ParamProvider(param_type=B2AKey))
        self.append_provider(ParamProvider(param_type=TanhKey))

    def send(self, x):
        self.communicator.send_to_address(self.target_address, x)

    def receive(self):
        ret = self.communicator.recv_from_address(self.receiving_address)
        if hasattr(ret, 'party'):
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
        self.communicator.close()
        for provider_thread in self.provider_threads.values():
            provider_thread.join()
        import os
        os.kill(os.getpid(), 0)


class SemiHonest3PCParty(Party3PC):
    """
    3方计算的参与者。

    此类包含多个功能，允许参与者与其他参与者进行通信，发送和接收数据，并加载和使用密钥和掩码。
    """

    def __init__(self, id):
        """初始化PartyMPC的一个实例。

        :param id: int,此参与者的ID。
        """
        super(SemiHonest3PCParty, self).__init__(id)
        if id != 2:
            self.set_comparison_provider()

    def set_comparison_provider(self):
        # pass
        if self.party_id != 2:
            self.append_provider(ParamProvider(SigmaDICFKey))
        # self.append_provider(GrottoDICFKey)  # EQUAL

    def set_trunc_provider(self):
        self.append_provider(ParamProvider(param_type=RssTruncAuxParams))

    def set_providers(self):
        self.set_trunc_provider()
        self.load_aux_params()
        for provider_thread in self.provider_threads.values():
            provider_thread.start()
