import os
from NssMPC.common.ring import RingTensor
from NssMPC.crypto.aux_parameter import Parameter
from NssMPC.config.configs import SCALE, param_path


class RssTruncAuxParams(Parameter):
    def __init__(self):
        self.r = None
        self.r_hat = None
        self.size = 0

    def __iter__(self):
        return iter((self.r, self.r_hat))

    @staticmethod
    def gen(num_of_params):
        from NssMPC.crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
        from NssMPC.config.configs import HALF_RING
        r = RingTensor.random([num_of_params], down_bound=-HALF_RING // 3, upper_bound=HALF_RING // 3)
        r_hat = r // SCALE
        r_list = ReplicatedSecretSharing.share(r)
        r_hat_list = ReplicatedSecretSharing.share(r_hat)
        aux_params = []
        for i in range(3):
            param = RssTruncAuxParams()
            param.r = r_list[i].to('cpu')
            param.r_hat = r_hat_list[i].to('cpu')
            param.size = num_of_params
            aux_params.append(param)
        return aux_params

    @classmethod
    def gen_and_save(cls, num):
        aux_params = cls.gen(num)
        file_path = f"{param_path}{cls.__name__}/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range(3):
            file_name = f"RssTruncAuxParams_{i}.pth"
            aux_params[i].save(file_path, file_name)
        # print(file_path)
