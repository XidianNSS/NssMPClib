from NssMPC import RingTensor
from NssMPC.common.utils.common_utils import list_rotate
from NssMPC.config import DEVICE
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.vsigma_key import VSigmaKey
from NssMPC.crypto.primitives import ReplicatedSecretSharing
from NssMPC.crypto.primitives.function_secret_sharing.vsigma import VSigma

from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional import b2a
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import mac_check, \
    recon, MACKey


class MacCheckPart(object):
    def __init__(self):
        # 假设mac_key只有一个
        self.value = None
        self.mac_value = None
        self.mac_key = None

    def add_value(self, value, mac_value):
        if self.value is None:
            self.value = value.flatten()
            self.mac_value = mac_value.flatten()
        else:
            self.value = self.value.cat(value.flatten())
            self.mac_value = self.mac_value.cat(mac_value.flatten())

    def set_key(self, mac_key):
        self.mac_key = mac_key

    def check(self):
        mac_check(self.value, self.mac_value, self.mac_key)

    def clear(self):
        self.value = None
        self.mac_value = None
        self.mac_key = None


class MSBWithOS(object):
    # def __init__(self, party):
    #     self.self_rdx0 = None
    #     self.self_rdx1 = None
    #     self.k0 = None  # get from P+2
    #     self.k1 = None  # get from P+1
    #     self.party = party
    #     self.share_table = None
    #     self.mac_key = None
    #     self.check_part = MacCheckPart()
    #
    # def preprocess(self, num_of_keys):
    #     party = self.party
    #     k0, k1 = SigmaDICF.gen(num_of_keys)
    #     party.send_params_to((party.party_id + 1) % 3, k0.to_dic())
    #     party.send_params_to((party.party_id + 2) % 3, k1.to_dic())
    #     k0p2 = SigmaDICFKey.from_dic(party.receive_params_dict_from((party.party_id + 2) % 3))
    #     k1p1 = SigmaDICFKey.from_dic(party.receive_params_dict_from((party.party_id + 1) % 3))
    #
    #     ObliviousSelect.check_keys_and_r()
    #
    #     # self.self_rdx0 = k0.r_in.to('cpu')
    #     # self.self_rdx1 = k1.r_in.to('cpu')
    #     # self.k0 = k0p2.to('cpu')
    #     # self.k1 = k1p1.to('cpu')
    #     self.self_rdx0 = k0.r_in
    #     self.self_rdx1 = k1.r_in
    #     self.k0 = k0p2
    #     self.k1 = k1p1
    #     mac_key = ReplicatedSecretSharing.random(1, party)
    #     if self.party.party_id == 0:
    #         self.share_table = share(
    #             RingTensor(torch.tensor([0, 1], dtype=data_type, device=DEVICE), self.party.scale),
    #             self.party)
    #     else:
    #         self.share_table = receive_share_from(input_id=0, party=self.party)
    #     share_table = mul_with_out_trunc(self.share_table, mac_key)
    #     mac_check(self.share_table, share_table, mac_key)
    #     # self.share_table = self.share_table.cat(share_table).reshape((2, 2)).to('cpu')
    #     # self.mac_key = mac_key.to('cpu')
    #     self.share_table = self.share_table.cat(share_table).reshape((2, 2))
    #     self.mac_key = mac_key

    # def set_share_table(self, share_table, mac_key):
    #     # share_table中必须包含原始的table和table的mac值
    #     self.share_table = share_table
    #     self.mac_key = mac_key

    @staticmethod
    def msb_without_mac_check(x, share_table=None):
        shape = x.shape
        x = x.reshape(-1, 1)
        num = x.shape[0]
        party = x.party

        self_rdx1, _, _ = party.get_param(f'VSigmaKey_{party.party_id}_0', num)
        self_rdx0, _, _ = party.get_param(f'VSigmaKey_{party.party_id}_1', num)
        next_party = party.virtual_party_with_next
        pre_party = party.virtual_party_with_previous
        key_from_next = next_party.get_param(VSigmaKey, num)
        key_from_previous = pre_party.get_param(VSigmaKey, num)
        rdx1, k1, c1 = key_from_previous
        rdx0, k0, c0 = key_from_next

        rdx_list = [ReplicatedSecretSharing([self_rdx1.item, self_rdx0.item], party),
                    ReplicatedSecretSharing([RingTensor.convert_to_ring(0), rdx1.item], party),
                    ReplicatedSecretSharing([rdx0.item, RingTensor.convert_to_ring(0)], party)
                    ]

        rdx_list = list_rotate(rdx_list, party.party_id)

        delta0 = x + rdx_list[0]
        delta1 = x + rdx_list[1]
        delta2 = x + rdx_list[2]

        dt1_list = []
        dt2_list = []

        dt1_list.append(recon(delta1, 0))
        dt2_list.append(recon(delta2, 0))

        dt1_list.append(recon(delta2, 1))
        dt2_list.append(recon(delta0, 1))

        dt1_list.append(recon(delta0, 2))
        dt2_list.append(recon(delta1, 2))

        # dt1 = dt1_list[party.party_id].to(x.device)
        # dt2 = dt2_list[party.party_id].to(x.device)
        dt1 = dt1_list[party.party_id]
        dt2 = dt2_list[party.party_id]

        v1, pi1 = VSigma.eval(dt1, (k1.to(DEVICE), c1.to(DEVICE)), 0)
        v2, pi2 = VSigma.eval(dt2, (k0.to(DEVICE), c0.to(DEVICE)), 1)
        # if DEBUG_LEVEL == 2:
        #     v1, pi1 = SigmaDICF.eval(dt1, (self.k1[0].ver_dpf_key, self.k1[0].c), 1)
        #     v2, pi2 = SigmaDICF.eval(dt2, (self.k0[0].ver_dpf_key, self.k0[0].c), 0)
        # else:
        #     v1, pi1 = SigmaDICF.eval(dt1, (self.k1[:num].ver_dpf_key, self.k1[:num].c), 1)
        #     v2, pi2 = SigmaDICF.eval(dt2, (self.k0[:num].ver_dpf_key, self.k0[:num].c), 0)

        # if party.party_id == 2: # TODO: 验证VDPF(离线全域验证)
        #     party.send((party.party_id + 1) % 3, pi2)
        #     o_pi2 = party.receive((party.party_id + 1) % 3)
        #     check_is_all_element_equal(pi2, o_pi2)
        #
        #     party.send((party.party_id + 2) % 3, pi1)
        #     o_pi1 = party.receive((party.party_id + 2) % 3)
        #     check_is_all_element_equal(pi1, o_pi1)
        # else:
        #     party.send((party.party_id + 2) % 3, pi1)
        #     o_pi1 = party.receive((party.party_id + 2) % 3)
        #     check_is_all_element_equal(pi1, o_pi1)
        #
        #     party.send((party.party_id + 1) % 3, pi2)
        #     o_pi2 = party.receive((party.party_id + 1) % 3)
        #     check_is_all_element_equal(pi2, o_pi2)

        if party.party_id == 0:
            v1_a = b2a(v1, party.virtual_party_with_previous)
            v2_a = b2a(v2, party.virtual_party_with_next)
        elif party.party_id == 1:
            v1_a = b2a(v1, party.virtual_party_with_previous)
            v2_a = b2a(v2, party.virtual_party_with_next)
        elif party.party_id == 2:
            v2_a = b2a(v2, party.virtual_party_with_next)
            v1_a = b2a(v1, party.virtual_party_with_previous)
        else:
            raise Exception("Party id error!")
        # share_table = self.share_table.to(DEVICE)
        if share_table is None:
            share_table, mac_key, smac_table = x.party.get_param(MACKey, num)
        else:
            share_table = share_table[0]
            smac_table = share_table[1]
            mac_key = None

        # v1_a_0 = RingTensor.convert_to_ring(1) - v1_a
        # v2_a_0 = -v2_a
        #
        # v1 = share_table.replicated_shared_tensor[0][:share_table.shape[0] // 2][0] * v1_a_0 + \
        #      share_table.replicated_shared_tensor[0][:share_table.shape[0] // 2][1] * v1_a
        # v2 = share_table.replicated_shared_tensor[1][:share_table.shape[0] // 2][0] * v2_a_0 + \
        #      share_table.replicated_shared_tensor[1][:share_table.shape[0] // 2][1] * v2_a
        #
        # mac_v1 = share_table.replicated_shared_tensor[0][share_table.shape[0] // 2:][0] * v1_a_0 + \
        #          share_table.replicated_shared_tensor[0][share_table.shape[0] // 2:][1] * v1_a
        # mac_v2 = share_table.replicated_shared_tensor[1][share_table.shape[0] // 2:][0] * v2_a_0 + \
        #          share_table.replicated_shared_tensor[1][share_table.shape[0] // 2:][1] * v2_a
        v1 = (RingTensor.mul(share_table.item[0], v1_a.item)).sum(-1)
        v2 = (RingTensor.mul(share_table.item[1], v2_a.item)).sum(-1)

        mac_v1 = (smac_table.item[0] * v1_a.item).sum(-1)
        mac_v2 = (smac_table.item[1] * v2_a.item).sum(-1)

        v = v1 + v2
        mac_v = mac_v1 + mac_v2

        v = ReplicatedSecretSharing.reshare(v, party)
        mac_v = ReplicatedSecretSharing.reshare(mac_v, party)

        return v.reshape(shape), mac_v.reshape(shape), mac_key

    # def msb(self, x):
    #     v, mac_v, mac_key = self.msb_without_mac_check(x)
    #     mac_check(v, mac_v, mac_key)
    #     return v
    #
    # def msb_later_check(self, x):
    #     v, mac_v, mac_key = self.msb_without_mac_check(x)
    #     self.check_part.add_value(v, mac_v)
    #     self.check_part.set_key(mac_key)
    #     return v

    # def compare_select(self, a, b, share_table):
    #     """
    #     share_table中有两组元素
    #     a < b，选择第0组，a >= b，选择第1组
    #     :param a:
    #     :param b:
    #     :param share_table:
    #     :return:
    #     """
    #     x = a - b
    #     x = x.reshape(-1, 1)
    #     num = x.shape[0]
    #     party = self.party
    #
    #     if DEBUG_LEVEL == 2:
    #         rdx_list = [ReplicatedSecretSharing([self.self_rdx1[0], self.self_rdx0[0]], party),
    #                     ReplicatedSecretSharing([RingTensor.convert_to_ring(0), self.k1.r_in[0]], party),
    #                     ReplicatedSecretSharing([self.k0.r_in[0], RingTensor.convert_to_ring(0)], party)
    #                     ]
    #     else:
    #         rdx_list = [ReplicatedSecretSharing([self.self_rdx1[:num], self.self_rdx0[:num]], party).view(-1, 1),
    #                     ReplicatedSecretSharing([RingTensor.convert_to_ring(0), self.k1.r_in[:num]], party).view(-1, 1),
    #                     ReplicatedSecretSharing([self.k0.r_in[:num], RingTensor.convert_to_ring(0)], party).view(-1, 1)
    #                     ]
    #
    #     rdx_list = list_rotate(rdx_list, party.party_id)
    #
    #     delta0 = x + rdx_list[0]
    #     delta1 = x + rdx_list[1]
    #     delta2 = x + rdx_list[2]
    #
    #     dt1_list = []
    #     dt2_list = []
    #
    #     dt1_list.append(recon(delta1, 0))
    #     dt2_list.append(recon(delta2, 0))
    #
    #     dt1_list.append(recon(delta2, 1))
    #     dt2_list.append(recon(delta0, 1))
    #
    #     dt1_list.append(recon(delta0, 2))
    #     dt2_list.append(recon(delta1, 2))
    #
    #     dt1 = dt1_list[party.party_id]
    #     dt2 = dt2_list[party.party_id]
    #
    #     if DEBUG_LEVEL == 2:
    #         v1, pi1 = SigmaDICF.eval(dt1, (self.k1[0].ver_dpf_key, self.k1[0].c), 1)
    #         v2, pi2 = SigmaDICF.eval(dt2, (self.k0[0].ver_dpf_key, self.k0[0].c), 0)
    #     else:
    #         v1, pi1 = SigmaDICF.eval(dt1, (self.k1[:num].ver_dpf_key, self.k1[:num].c), 1)
    #         v2, pi2 = SigmaDICF.eval(dt2, (self.k0[:num].ver_dpf_key, self.k0[:num].c), 0)
    #
    #     # if party.party_id == 2:
    #     #     party.send((party.party_id + 1) % 3, pi2)
    #     #     o_pi2 = party.receive((party.party_id + 1) % 3)
    #     #     check_is_all_element_equal(pi2, o_pi2)
    #
    #     # if party.party_id == 2:
    #     #     party.send((party.party_id + 1) % 3, pi2)
    #     #     o_pi2 = party.receive((party.party_id + 1) % 3)
    #     #     check_is_all_element_equal(pi2, o_pi2)
    #     #
    #     #     party.send((party.party_id + 2) % 3, pi1)
    #     #     o_pi1 = party.receive((party.party_id + 2) % 3)
    #     #     check_is_all_element_equal(pi1, o_pi1)
    #     # else:
    #     #     party.send((party.party_id + 2) % 3, pi1)
    #     #     o_pi1 = party.receive((party.party_id + 2) % 3)
    #     #     check_is_all_element_equal(pi1, o_pi1)
    #     #
    #     #     party.send((party.party_id + 1) % 3, pi2)
    #     #     o_pi2 = party.receive((party.party_id + 1) % 3)
    #     #     check_is_all_element_equal(pi2, o_pi2)
    #
    #     if party.party_id == 0:
    #         # todo: 恶意检测？
    #         v1_a = b2a(v1, party.virtual_party_with_previous)
    #         v2_a = b2a(v2, party.virtual_party_with_next)
    #     elif party.party_id == 1:
    #         v1_a = b2a(v1, party.virtual_party_with_previous)
    #         v2_a = b2a(v2, party.virtual_party_with_next)
    #     elif party.party_id == 2:
    #         v2_a = b2a(v2, party.virtual_party_with_next)
    #         v1_a = b2a(v1, party.virtual_party_with_previous)
    #     else:
    #         raise Exception("Party id error!")
    #
    #     v1_a_0 = RingTensor.convert_to_ring(1) - v1_a
    #     v2_a_0 = -v2_a
    #
    #     v1 = share_table.replicated_shared_tensor[0][:share_table.shape[0] // 2, 0].reshape(-1, 1) * v1_a + \
    #          share_table.replicated_shared_tensor[0][:share_table.shape[0] // 2, 1].reshape(-1, 1) * v1_a_0
    #     v2 = share_table.replicated_shared_tensor[1][:share_table.shape[0] // 2, 0].reshape(-1, 1) * v2_a + \
    #          share_table.replicated_shared_tensor[1][:share_table.shape[0] // 2, 1].reshape(-1, 1) * v2_a_0
    #
    #     mac_v1 = share_table.replicated_shared_tensor[0][share_table.shape[0] // 2:, 0].reshape(-1, 1) * v1_a + \
    #              share_table.replicated_shared_tensor[0][share_table.shape[0] // 2:, 1].reshape(-1, 1) * v1_a_0
    #     mac_v2 = share_table.replicated_shared_tensor[1][share_table.shape[0] // 2:, 0].reshape(-1, 1) * v2_a + \
    #              share_table.replicated_shared_tensor[1][share_table.shape[0] // 2:, 1].reshape(-1, 1) * v2_a_0
    #
    #     v = v1 + v2
    #     mac_v = mac_v1 + mac_v2
    #
    #     v = ReplicatedSecretSharing.reshare(v, party)
    #     mac_v = ReplicatedSecretSharing.reshare(mac_v, party)
    #
    #     # 这里不做mac_check，最后统一做
    #     return v.reshape((num, -1)), mac_v.reshape((num, -1))
