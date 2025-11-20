# #  This file is part of the NssMPClib project.
# #  Copyright (c) 2024 XDU NSS lab,
# #  Licensed under the MIT license. See LICENSE in the project root for license information.
#
# from NssMPC.protocols.honest_majority_3pc.msb_with_os import MACKey
# from NssMPC.protocols.honest_majority_3pc.multiplication import RssMulTriples, RssMatmulTriples, RssMatrixBeaverProvider
# from NssMPC.protocols.semi_honest_2pc.b2a import B2AKey
# from NssMPC.primitives.secret_sharing.function import VSigmaKey
# from NssMPC.protocols.honest_majority_3pc.oblivious_select_dpf import VOSKey
# from NssMPC.protocols.honest_majority_3pc.truncate import RssTruncAuxParams
# from NssMPC.infra.mpc.party.party import Party3PC
# from NssMPC.infra.mpc.param_provider import ParamProvider
#
#
# class HonestMajorityParty(Party3PC):
#     """
#     Participants in 3-party computation. This class contains several functionalities, allowing participants to communicate with each other, send and receive data, and load and use keys and masks.
#     """
#
#     def __init__(self, id):
#         """
#         Initializes an instance of PartyMPC.
#
#         :param id: The ID of this participant
#         :type id: int
#         """
#         super(HonestMajorityParty, self).__init__(id)
#
#     def load_aux_params(self):
#         """
#         Load auxiliary parameters.
#
#         Iterate through all providers of the current participant, loading the corresponding parameters according to the different parameter types:
#             * If it is RssMatmulTriples, no processing is performed.
#             * if it is VOSKey or VSigmaKey, the parameters are loaded based on the provider's type
#
#         If the provider's parameter type is not of some particular type, the corresponding parameter file is loaded,
#         and the party attribute of the secondary parameter is set to the same as ``self``.Finally,
#         the :meth:`~NssMPC.runtime.party.party.VirtualParty2PC.load_aux_params` method is recursively called
#         before and after the virtual participant.
#
#         """
#         for name, provider in self.providers.items():
#             if provider.param_type == RssMatmulTriples:
#                 continue
#             if provider.param_type == VOSKey or provider.param_type == VSigmaKey:
#                 provider.load_param()
#                 continue
#             param_saved_name = provider.saved_name + '_' + str(self.party_id) + '.pth'
#             provider.load_param(saved_name=param_saved_name)
#         self.virtual_party_with_previous.load_aux_params()
#         self.virtual_party_with_next.load_aux_params()
#
#
#     def set_multiplication_provider(self):
#         """
#         Set up multiplication provider.
#
#         Use the key RssMulTriples to create a ParamProvider and a RssMatrixBeaverProvider to provide multiplication of key providers for participants.
#         """
#         self.append_provider(ParamProvider(RssMulTriples))
#
#         self.set_trunc_provider()
#
#     def set_trunc_provider(self):
#         """
#         Set up truncation provider.
#
#         Use the key RssTruncAuxParams to create a ParamProvider to truncate.
#         """
#         self.append_provider(ParamProvider(RssTruncAuxParams))
