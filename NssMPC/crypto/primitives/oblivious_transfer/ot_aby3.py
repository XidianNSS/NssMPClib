"""
参考ABY3的3方2选1OT实现
"""

from NssMPC import RingTensor

from NssMPC.config import DEVICE


class OT(object):
    @staticmethod
    def sender(m0, m1, party, receiver_id, helper_id):
        """
        3方OT模型下的sender
        :param m0:sender所持有的信息m0
        :param m1:sender所持有的信息m1
        :param party:当前参与方
        :param receiver_id:接收方的编号
        :param helper_id:帮助方的编号
        :return:
        """

        w0 = RingTensor.random(m0.shape, 'int', DEVICE, 0, 2, )

        party.send(helper_id, w0)
        w1 = party.receive(helper_id)

        m0 = m0 ^ w0
        m1 = m1 ^ w1
        party.send(receiver_id, m0)
        party.send(receiver_id, m1)

    @staticmethod
    def receiver(c, party, sender_id, helper_id):
        """
        三方OT模型下的receiver helper默认位置在party_id + 1
        receiver需要选择wc
        :param c:选择位
        :param party:当前参与方
        :param sender_id:发送方的编号
        :param helper_id:帮助方的编号
        :return:
        """

        m0_masked = party.receive(sender_id)
        m1_masked = party.receive(sender_id)
        wc = party.receive(helper_id)
        cond = (c > 0) + 0
        mc = m0_masked * (1 - cond) + m1_masked * cond
        mc = mc ^ wc

        return mc

    @staticmethod
    def helper(c, party, sender_id, receiver_id):
        """
        三方OT模型下的helper
        helper知道receiver需要选择wc
        :param party:当前参与方
        :param c:选择位
        :param sender_id:发送方的编号
        :param receiver_id:接收方的编号
        :return:
        """

        w0 = party.receive(sender_id)
        w1 = RingTensor.random(w0.shape, 'int', DEVICE, 0, 2, )
        party.send(sender_id, w1)

        cond = (c > 0) + 0

        mc = w0 * (1 - cond) + w1 * cond

        party.send(receiver_id, mc)
