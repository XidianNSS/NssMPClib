import torch

from NssMPC import RingTensor
from NssMPC.config import data_type
from NssMPC.crypto.aux_parameter import Parameter


class Wrap(Parameter):
    def __init__(self, r=None, theta_r=None):
        self.r = r
        self.theta_r = theta_r

    @staticmethod
    def gen(num_of_params):
        from NssMPC.crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing
        r = RingTensor.random([num_of_params])
        r0, r1 = ArithmeticSecretSharing.share(r, 2)
        theta_r = Wrap.count_wraps([r0.item.tensor, r1.item.tensor])

        theta_r0, theta_r1 = ArithmeticSecretSharing.share(RingTensor(theta_r), 2)

        wrap_0 = Wrap(r0.item.tensor, theta_r0.item.tensor)
        wrap_1 = Wrap(r1.item.tensor, theta_r1.item.tensor)

        return wrap_0, wrap_1

    @staticmethod
    def count_wraps(share_list):
        """Computes the number of overflows or underflows in a set of shares

        We compute this by counting the number of overflows and underflows as we
        traverse the list of shares.
        """
        result = torch.zeros_like(share_list[0], dtype=data_type)
        prev = share_list[0]
        for cur in share_list[1:]:
            next = cur + prev
            result -= ((prev < 0) & (cur < 0) & (next > 0)).to(data_type)  # underflow
            result += ((prev > 0) & (cur > 0) & (next < 0)).to(data_type)  # overflow
            prev = next
        return result
