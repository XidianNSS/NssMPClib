from NssMPC import RingTensor
from NssMPC.crypto.aux_parameter import Parameter


class MACKey(Parameter):
    def __init__(self, share_table=None, mac_key=None, smac_table=None):
        self.share_table = share_table
        self.mac_key = mac_key
        self.smac_table = smac_table

    def __iter__(self):
        return iter([self.share_table, self.mac_key, self.smac_table])

    @staticmethod
    def gen(num_of_keys):
        table1 = RingTensor.ones([num_of_keys, 1])

        from NssMPC import ReplicatedSecretSharing
        share_tables = ReplicatedSecretSharing.share(table1)

        mac_key = RingTensor.random([num_of_keys, 1])
        share_keys = ReplicatedSecretSharing.share(mac_key)

        mac_table = table1 * mac_key
        smac_tables = ReplicatedSecretSharing.share(mac_table)

        return [MACKey(a, b, c) for a, b, c in zip(share_tables, share_keys, smac_tables)]
