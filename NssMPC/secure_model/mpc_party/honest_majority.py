from NssMPC.crypto.aux_parameter import RssMulTriples, RssMatmulTriples, B2AKey
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.vsigma_key import VSigmaKey
from NssMPC.crypto.aux_parameter.select_keys.vos_key import VOSKey
from NssMPC.crypto.aux_parameter.truncation_keys.rss_trunc_aux_param import RssTruncAuxParams
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.comparison import MaliciousCMPKey, \
    MACKey
from NssMPC.secure_model.mpc_party.party import Party3PC
from NssMPC.secure_model.utils.param_provider.matrix_beaver_provider import RssMatrixBeaverProvider
from NssMPC.secure_model.utils.param_provider.os_provider import OSProvider
from NssMPC.secure_model.utils.param_provider.param_provider import ParamProvider


class HonestMajorityParty(Party3PC):
    """
    3方计算的参与者。

    此类包含多个功能，允许参与者与其他参与者进行通信，发送和接收数据，并加载和使用密钥和掩码。
    """

    def __init__(self, id):
        """初始化PartyMPC的一个实例。

        :param id: int,此参与者的ID。
        """
        super(HonestMajorityParty, self).__init__(id)

    def load_aux_params(self):
        for name, provider in self.providers.items():
            if provider.param_type == RssMatmulTriples:
                continue
            if provider.param_type == VOSKey or provider.param_type == VSigmaKey:
                provider.load_param()
                if hasattr(provider.param, 'set_party'):
                    provider.param.set_party(self)
                continue
            param_saved_name = provider.saved_name + '_' + str(self.party_id) + '.pth'
            provider.load_param(saved_name=param_saved_name)
            if hasattr(provider.param, 'set_party'):
                provider.param.set_party(self)
        self.virtual_party_with_previous.load_aux_params()
        self.virtual_party_with_next.load_aux_params()

    def set_comparison_provider(self):
        self.append_provider(ParamProvider(MACKey))
        self.virtual_party_with_previous.append_provider(ParamProvider(MaliciousCMPKey))
        self.virtual_party_with_next.append_provider(ParamProvider(MaliciousCMPKey))

    def set_oblivious_selection_provider(self):
        self.virtual_party_with_previous.append_provider(
            OSProvider(VOSKey, saved_name=f'VOSKey_{(self.party_id + 1) % 3}_0'))
        self.virtual_party_with_next.append_provider(
            OSProvider(VOSKey, saved_name=f'VOSKey_{(self.party_id + 2) % 3}_1'))
        self.append_provider(
            OSProvider(VOSKey, saved_name=f'VOSKey_{self.party_id}_0', param_tag=f'VOSKey_{self.party_id}_0'))
        self.append_provider(
            OSProvider(VOSKey, saved_name=f'VOSKey_{self.party_id}_1', param_tag=f'VOSKey_{self.party_id}_1'))

    def set_conditional_oblivious_selection_provider(self):
        self.virtual_party_with_previous.append_provider(
            OSProvider(VSigmaKey, saved_name=f'VSigmaKey_{(self.party_id + 1) % 3}_0'))
        self.virtual_party_with_next.append_provider(
            OSProvider(VSigmaKey, saved_name=f'VSigmaKey_{(self.party_id + 2) % 3}_1'))
        self.append_provider(
            OSProvider(VSigmaKey, saved_name=f'VSigmaKey_{self.party_id}_0', param_tag=f'VSigmaKey_{self.party_id}_0'))
        self.append_provider(
            OSProvider(VSigmaKey, saved_name=f'VSigmaKey_{self.party_id}_1', param_tag=f'VSigmaKey_{self.party_id}_1'))
        self.virtual_party_with_previous.append_provider(ParamProvider(B2AKey))
        self.virtual_party_with_next.append_provider(ParamProvider(B2AKey))
        self.append_provider(ParamProvider(MACKey))

    def set_multiplication_provider(self):
        self.append_provider(ParamProvider(RssMulTriples))
        self.append_provider(RssMatrixBeaverProvider(party=self))
        # self.append_provider(Wrap)

    def set_trunc_provider(self):
        self.append_provider(ParamProvider(RssTruncAuxParams))
