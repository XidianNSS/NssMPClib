from NssMPC.crypto.aux_parameter import Parameter, RssMatmulTriples
from NssMPC.config.configs import SCALE
from NssMPC.common.ring import *
from NssMPC.crypto.aux_parameter.truncation_keys.rss_trunc_aux_param import RssTruncAuxParams
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.base import open


def truncate_preprocess_online(share, scale=SCALE):
    # TODO:临时使用的方法
    # 这里直接生成随机数r 直接限制其生成范围使执行recon时不超环，直接在本地进行trunc
    party = share.party
    from NssMPC.crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
    from NssMPC.config.configs import HALF_RING
    r_hat_local = RingTensor.random(share.shape, down_bound=-HALF_RING, upper_bound=HALF_RING)
    r_local = r_hat_local // scale
    r_local.dtype = 'int'  # 在换上进行trunc计算，当成int来看
    r_hat = ReplicatedSecretSharing.share(r_hat_local)
    r_hat.dtype = 'int'
    r = ReplicatedSecretSharing.share(r_local)
    return r_hat, r


def truncate_online(share, scale=SCALE):
    # 使用ABY3的trunc2进行计算
    # 该方法的核心思想是使用一个预先计算的r和其trunc值r'对秘密分享的数据进行盲化，然后恢复成明文 z = x + r，随后再进行计算
    # z' = trunc(z) 最后再用秘密分享的r'对其进行恢复，得到 x' = z' - r'
    # 这一计算步骤使得截断操作发生在明文上，因此不用考虑秘密分享恢复的超环问题，但是仍然存在 x+r的超环问题
    # 因此本实现限定了r的取值范围和x的值域范围 使得 x+r < RingMax
    r, r_t = truncate_preprocess_online(share, scale)
    delta_share = share - r
    delta = open(delta_share)
    delta_trunc = delta // scale
    result = r_t + delta_trunc
    return result


def truncate(share):
    # 使用ABY3的trunc2进行计算
    # 该方法的核心思想是使用一个预先计算的r和其trunc值r'对秘密分享的数据进行盲化，然后恢复成明文 z = x + r，随后再进行计算
    # z' = trunc(z) 最后再用秘密分享的r'对其进行恢复，得到 x' = z' - r'
    # 这一计算步骤使得截断操作发生在明文上，因此不用考虑秘密分享恢复的超环问题，但是仍然存在 x+r的超环问题
    # 因此本实现限定了r的取值范围和x的值域范围 使得 x+r < RingMax
    r, r_t = share.party.get_param(RssTruncAuxParams, share.numel())
    shape = share.shape
    share = share.flatten()
    r.party = share.party
    r_t.party = share.party
    # todo:计算原理需要统一
    r_t.dtype = 'float'
    r.dtype = 'float'
    delta_share = share - r
    delta = open(delta_share)
    delta_trunc = delta // SCALE
    result = r_t + delta_trunc
    return result.reshape(shape)
