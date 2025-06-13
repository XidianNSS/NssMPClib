#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC.crypto.aux_parameter import RssMulTriples, RssMatmulTriples, B2AKey, MACKey
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.vsigma_key import VSigmaKey
from NssMPC.crypto.aux_parameter.select_keys.vos_key import VOSKey
from NssMPC.crypto.aux_parameter.truncation_keys.rss_trunc_aux_param import RssTruncAuxParams
from NssMPC.secure_model.mpc_party.party import Party3PC
from NssMPC.secure_model.utils.param_provider.matrix_beaver_provider import RssMatrixBeaverProvider
from NssMPC.secure_model.utils.param_provider.os_provider import OSProvider
from NssMPC.secure_model.utils.param_provider.param_provider import ParamProvider


class HonestMajorityParty(Party3PC):
    """
    Participants in 3-party computation. This class contains several functionalities, allowing participants to communicate with each other, send and receive data, and load and use keys and masks.
    """

    def __init__(self, id):
        """
        Initializes an instance of PartyMPC.

        :param id: The ID of this participant
        :type id: int
        """
        super(HonestMajorityParty, self).__init__(id)

    def load_aux_params(self):
        """
        Load auxiliary parameters.

        Iterate through all providers of the current participant, loading the corresponding parameters according to the different parameter types:
            * If it is RssMatmulTriples, no processing is performed.
            * if it is VOSKey or VSigmaKey, the parameters are loaded based on the provider's type

        If the provider's parameter type is not of some particular type, the corresponding parameter file is loaded,
        and the party attribute of the secondary parameter is set to the same as ``self``.Finally,
        the :meth:`~NssMPC.secure_model.mpc_party.party.VirtualParty2PC.load_aux_params` method is recursively called
        before and after the virtual participant.

        """
        for name, provider in self.providers.items():
            if provider.param_type == RssMatmulTriples:
                continue
            if provider.param_type == VOSKey or provider.param_type == VSigmaKey:
                provider.load_param()
                continue
            param_saved_name = provider.saved_name + '_' + str(self.party_id) + '.pth'
            provider.load_param(saved_name=param_saved_name)
        self.virtual_party_with_previous.load_aux_params()
        self.virtual_party_with_next.load_aux_params()

    def set_oblivious_selection_provider(self):
        """
        Set up the blind selection provider. Provide blind selection of key providers for participants and their nearby virtual participants.

        The blind selection provider is created using VOSKey as the key for the forward and backward participants and passed to the :meth:`~NssMPC.secure_model.mpc_party.party.Party.append_provider` method as the parameter.
        The current participant creates blind selection provider_0 and blind selection provider_1 in the same way.
        """
        self.virtual_party_with_previous.append_provider(
            OSProvider(VOSKey, saved_name=f'VOSKey_{(self.party_id + 1) % 3}_0'))
        self.virtual_party_with_next.append_provider(
            OSProvider(VOSKey, saved_name=f'VOSKey_{(self.party_id + 2) % 3}_1'))
        self.append_provider(
            OSProvider(VOSKey, saved_name=f'VOSKey_{self.party_id}_0', param_tag=f'VOSKey_{self.party_id}_0'))
        self.append_provider(
            OSProvider(VOSKey, saved_name=f'VOSKey_{self.party_id}_1', param_tag=f'VOSKey_{self.party_id}_1'))

    def set_comparison_provider(self):
        """
        Set up conditional oblivious selection provider.

        First, set the Sigma key provider:
            The conditional oblivious selection provider is created using VSigmaKey as the key for the forward and backward participants and passed to the :meth:`~NssMPC.secure_model.mpc_party.party.Party.append_provider` method as the parameter.
            The current participant creates conditional oblivious selection provider_0 and conditional oblivious selection provider_1 in the same way.

        Then, set the B2A key provider:
            The conditional oblivious selection provider is created using VOSKey as the key for the forward and
            backward participants and passed to the :meth:`~NssMPC.secure_model.mpc_party.party.Party.append_provider`
            method as the parameter.

        Finally, set the MAC key provider:
            The conditional oblivious selection provider is created using MACKey as the key for the forward and
            backward participants and passed to the :meth:`~NssMPC.secure_model.mpc_party.party.Party.append_provider`
            method as the parameter.

        """
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
        """
        Set up multiplication provider.

        Use the key RssMulTriples to create a ParamProvider and a RssMatrixBeaverProvider to provide multiplication of key providers for participants.
        """
        self.append_provider(ParamProvider(RssMulTriples))
        self.append_provider(RssMatrixBeaverProvider(party=self))
        self.set_trunc_provider()

    def set_trunc_provider(self):
        """
        Set up truncation provider.

        Use the key RssTruncAuxParams to create a ParamProvider to truncate.
        """
        self.append_provider(ParamProvider(RssTruncAuxParams))
