"""
This document defines the OT class for 1-out-of-2 OT(Oblivious Transfer) among 3 parties.
The implementation is based on the work of
W. Peter, X. Samee, R. Xiao, "ABY3: A Mixed Protocol Framework for Machine Learning," in IACR Cryptology ePrint Archive, Report 2018/403, 2018.
For reference, see the `paper <https://eprint.iacr.org/2018/403.pdf>`_.
"""

#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC import RingTensor

from NssMPC.config import DEVICE


class OT(object):
    """
    This class is used to implement 3-party 1-out-of-2 Oblivious Transfer (OT).

    We use this class to implement 1-out-of-2 OT, where the functions sender, receiver, and helper are defined.
    Through the computation and communication of the three parties, the protocol is executed.
    """

    @staticmethod
    def sender(m0, m1, party, receiver_id, helper_id):
        """
        Sender in the 3-party OT protocol

        :param m0: message m0 held by the sender
        :type m0: RingTensor
        :param m1: message m1 held by the sender
        :type m1: RingTensor
        :param party: current party involved
        :type party: Party
        :param receiver_id: ID of the receiver
        :type receiver_id: int
        :param helper_id: ID of the helper
        :type helper_id: int
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
        Receiver in the 3-party OT protocol
        By default, the helper is positioned at party_id + 1

        .. hint::
            The receiver needs to choose wc.

        :param c: selection bit, 0 or 1
        :type c: RingTensor
        :param party: current party involved
        :type party: Party
        :param sender_id: ID of the sender
        :type sender_id: int
        :param helper_id: ID of the helper
        :type helper_id: int
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
        Helper in the 3-party OT protocol

        .. hint::
            The helper knows the receiver needs to choose wc.

        :param c: selection bit, 0 or 1
        :type c: RingTensor
        :param party: current party involved
        :type party: Party
        :param sender_id: ID of the sender
        :type sender_id: int
        :param receiver_id: ID of the receiver
        :type receiver_id: int
        """

        w0 = party.receive(sender_id)
        w1 = RingTensor.random(w0.shape, 'int', DEVICE, 0, 2, )
        party.send(sender_id, w1)

        cond = (c > 0) + 0

        mc = w0 * (1 - cond) + w1 * cond

        party.send(receiver_id, mc)
