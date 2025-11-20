#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
#
import contextvars
from contextlib import contextmanager

from NssMPC.infra.mpc.party import Party, PartyCtx
from NssMPC.primitives.secret_sharing import AdditiveSecretSharing, BooleanSecretSharing, ReplicatedSecretSharing
from NssMPC.runtime.presets import SEMI_HONEST

_party_stack = contextvars.ContextVar('party_stack', default=[])


@contextmanager
def PartyRuntime(party:Party):
    """A runtime manager that supports nested contexts.

    This version properly handles nested contexts by maintaining a stack.

    Args:
        party: The party instance to set as current.
        protocol_cfg: Optional protocol configuration dictionary.
    """

    # 保存当前上下文token
    if prev_party := PartyCtx.get():
        party_stack = _party_stack.get()
        party_stack.append(prev_party)
        _party_stack.set(party_stack)
    PartyCtx.set(party)
    configure_mpc_protocols(party.thread_model_cfg)
    try:
        yield
    finally:
        if party_stack := _party_stack.get():
            prev_party = party_stack.pop()
            PartyCtx.set(prev_party)
            _party_stack.set(party_stack)
            configure_mpc_protocols(PartyCtx.get().thread_model_cfg)
        else:
            PartyCtx.set(None)


def configure_mpc_protocols(config: dict = SEMI_HONEST):
    """
    Configures the MPC protocols by dynamically assigning protocol implementations
    to the corresponding secret sharing classes based on the provided configuration.
    """
    scheme_classes = {
        'additive': AdditiveSecretSharing,
        'boolean': BooleanSecretSharing,
        'replicated': ReplicatedSecretSharing
    }

    for scheme_name, scheme_class in scheme_classes.items():
        if scheme_name in config:
            for attr, protocol in config[scheme_name].items():
                setattr(scheme_class, attr, protocol)
