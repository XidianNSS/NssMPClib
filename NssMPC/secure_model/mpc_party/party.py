import random
import time
from multiprocessing import Pipe, Lock

from NssMPC.common.random.prg import MT19937_PRG
from NssMPC.config import DEBUG_LEVEL, BIT_LEN, SOCKET_NUM, SOCKET_TYPE
from NssMPC.crypto.aux_parameter import MatmulTriples, RssMatmulTriples
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import MaliciousCMPKey
from NssMPC.secure_model.utils.buffer_thread.buffer_thread import BufferThread

if SOCKET_TYPE == 0:
    from NssMPC.common.network.communicator import NCommunicator as Communicator
else:
    from NssMPC.common.network.communicator import MCommunicator as Communicator


class Party(object):
    """
    A class representing a participating party.
    Each Party object has a unique party_id to identify it.

    Attributes:
        party_id (int): unique identifier representing the party
    """

    def __init__(self, party_id):
        self.communicator = Communicator()
        self.party_id = party_id
        self.providers = {}
        self.provider_threads = {}

    def load_aux_params(self):
        for name, provider in self.providers.items():
            if provider.param_type == MatmulTriples:
                continue
            if provider.param_type == MaliciousCMPKey:
                continue
            if provider.param_type == RssMatmulTriples:
                continue
            param_saved_name = provider.saved_name + '_' + str(self.party_id) + '.pth'
            provider.load_param(saved_name=param_saved_name)
            if hasattr(provider.param, 'set_party'):
                provider.param.set_party(self)

    def append_provider(self, provider):
        """
        这个函数仅作为给Party中添加provider的功能，此时provider必须被初始化，且需要指定名称
        Args:
            provider: provider实例

        Returns:
        """
        # append不再使用类名做字典的键值，避免一个party无法载入两个相同的Parameter的问题（3方两两执行22分享的计算场景）
        self.providers[provider.param_tag] = provider
        if not DEBUG_LEVEL:
            self.provider_threads[provider.param_tag] = BufferThread(provider, self, pipe=Pipe(True), lock=Lock())

    def get_param(self, param_tag, *args):
        """
        Get the parameter of a certain type.
        *args: the arguments needed to get the parameter. The number of arguments is determined by the type of parameter.
        """

        if isinstance(param_tag, str):
            return self.providers[param_tag].get_parameters(*args)

        else:
            return self.providers[param_tag.__name__].get_parameters(*args)


class Party3PC(Party):

    def __init__(self, party_id):
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
        """ 检查所有的参与者是否在线。
        :return:
        """
        time.sleep(3)
        num = 2 * SOCKET_NUM if SOCKET_TYPE else 2
        while self.communicator.connected_num() < num:
            # print(self.communicator.connected_num())
            pass

    def generate_prg_seed(self):
        prg_seed_0 = random.randint(0, 2 ** BIT_LEN)
        self.send((self.party_id - 1) % 3, prg_seed_0)
        prg_seed_1 = self.receive((self.party_id + 1) % 3)
        self.prg_0 = MT19937_PRG()
        self.prg_0.set_seeds(prg_seed_0)
        self.prg_1 = MT19937_PRG()
        self.prg_1.set_seeds(prg_seed_1)

    def set_communicator_address(self, address):
        assert isinstance(address, tuple)
        self.communicator.set_address(address[0], address[1])

    def send(self, target_id, x):
        self.communicator.send_to_address(self.target_address_mapping[target_id], x)

    def receive(self, target_id):
        ret = self.communicator.recv_from_address(self.receiving_address_mapping[target_id])
        from NssMPC.crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing
        from NssMPC.crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
        if isinstance(ret, (ArithmeticSecretSharing, ReplicatedSecretSharing)):
            ret.party = self
        return ret

    def close(self):
        """
        Close the connection of tcp
        """
        self.communicator.close()
        # for provider_thread in self.provider_threads.values():
        #     provider_thread.join()
        import os
        os.kill(os.getpid(), 0)


class VirtualParty2PC(Party):
    def __init__(self, party_id, real_party):
        super().__init__(party_id)
        self.real_party = real_party
        self.other_id = (real_party.party_id - party_id - 1) % 3

    def load_aux_params(self):
        for name, provider in self.providers.items():
            kid = (self.real_party.party_id + 1 + self.party_id) % 3
            param_saved_name = f'{provider.saved_name}_{kid}_{self.party_id}.pth'
            if name == 'VOSKey' or name == 'VSigmaKey':
                param_saved_name = None
            provider.load_param(saved_name=param_saved_name)
            if hasattr(provider.param, 'set_party'):
                provider.param.set_party(self)

    def send(self, x):
        self.real_party.send(self.other_id, x)

    def receive(self):
        ret = self.real_party.receive(self.other_id)
        if hasattr(ret, 'party'):
            ret.party = self
        return ret
