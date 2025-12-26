"""
This document defines the OT class for 1-out-of-2 OT(Oblivious Transfer) among 3 parties.
The implementation is based on the work of
W. Peter, X. Samee, R. Xiao, "ABY3: A Mixed Protocol Framework for Machine Learning," in IACR Cryptology ePrint Archive, Report 2018/403, 2018.
For reference, see the `paper <https://eprint.iacr.org/2018/403.pdf>`_.
"""

#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from nssmpc.config import DEVICE
from nssmpc.infra.tensor import RingTensor


class OT(object):
    """
    This class implements 3-party 1-out-of-2 Oblivious Transfer (OT).

    It defines the roles of sender, receiver, and helper to execute the protocol through computation and communication.
    The implementation is based on the ABY3 framework.

    References:
        W. Peter, X. Samee, R. Xiao, "ABY3: A Mixed Protocol Framework for Machine Learning," in IACR Cryptology ePrint Archive, Report 2018/403, 2018.
        `Paper <https://eprint.iacr.org/2018/403.pdf>`_
    """

    @staticmethod
    def sender(m0, m1, party, receiver_id, helper_id):
        """
        The sender role in the 3-party OT protocol.

        Args:
            m0 (RingTensor): The first message held by the sender.
            m1 (RingTensor): The second message held by the sender.
            party (Party): The current party instance involved in the protocol.
            receiver_id (int): The ID of the receiver party.
            helper_id (int): The ID of the helper party.

        Examples:
            >>> OT.sender(m0, m1, party, receiver_id=1, helper_id=2)
        """

        w0 = RingTensor.random(m0.shape, 'int', DEVICE, 0, 2, )

        party.send(helper_id, w0)
        w1 = party.recv(helper_id)

        m0 = m0 ^ w0
        m1 = m1 ^ w1
        party.send(receiver_id, m0)
        party.send(receiver_id, m1)

    @staticmethod
    def receiver(c, party, sender_id, helper_id):
        """
        The receiver role in the 3-party OT protocol.

        By default, the helper is positioned at party_id + 1. The receiver selects the message based on the bit `c`.

        Args:
            c (RingTensor): The selection bit (0 or 1).
            party (Party): The current party instance involved in the protocol.
            sender_id (int): The ID of the sender party.
            helper_id (int): The ID of the helper party.

        Returns:
            RingTensor: The selected message `mc`.

        Examples:
            >>> mc = OT.receiver(c, party, sender_id=0, helper_id=2)
        """

        m0_masked = party.recv(sender_id)
        m1_masked = party.recv(sender_id)
        wc = party.recv(helper_id)
        cond = c > 0
        mc = m0_masked * (1 - cond) + m1_masked * cond
        mc = mc ^ wc

        return mc

    @staticmethod
    def helper(c, party, sender_id, receiver_id):
        """
        The helper role in the 3-party OT protocol.

        The helper assists the receiver in obtaining the correct message without learning the selection bit.

        Args:
            c (RingTensor): The selection bit (0 or 1), known to the helper in this context (shared randomness).
            party (Party): The current party instance involved in the protocol.
            sender_id (int): The ID of the sender party.
            receiver_id (int): The ID of the receiver party.

        Examples:
            >>> OT.helper(c, party, sender_id=0, receiver_id=1)
        """

        w0 = party.recv(sender_id)
        w1 = RingTensor.random(w0.shape, 'int', DEVICE, 0, 2, )
        party.send(sender_id, w1)

        cond = (c > 0) + 0

        mc = w0 * (1 - cond) + w1 * cond

        party.send(receiver_id, mc)
