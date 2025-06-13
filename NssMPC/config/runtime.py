#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

class Runtime:
    """
    A class to manage the runtime environment of the NssMPClib library.
    Only supports `with runtime(party):` syntax.
    """
    _current_party = None
    _party_stack = []

    def __call__(self, party):
        """
        Set the current party for the runtime context.
        This method allows the runtime to be called with a specific party, which will be used in the context manager.
        :param party: The party to set as the current runtime context.
        :type party: Party
        :return: The current instance of the Runtime class.
        """
        if self._current_party is not None:
            self._party_stack.append(self._current_party)
        self._current_party = party
        return self

    def __enter__(self):
        """
        Enter the runtime context.
        This method is called when entering the `with` statement. It ensures that a party has been set before entering the context.
        """
        if self._current_party is None:
            raise RuntimeError("Runtime must be called with a party before entering the context.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context, restoring the previous party.
        This method is called when exiting the `with` statement. It restores the previous party from the stack if available.
        """
        if self._party_stack:
            self._current_party = self._party_stack.pop()
        else:
            self._current_party = None

    @property
    def party(self):
        """
        Get the current party in the runtime context.
        This property returns the current party that has been set in the runtime context.
        :return: The current party.
        :rtype: Party
        """
        if self._current_party is None:
            raise RuntimeError("No party is currently set in the runtime context.")
        return self._current_party


PartyRuntime = Runtime()


class MacBuffer:
    """
    Manage the operation of secret shared values and validate them with message verification codes

    This class provides the buffer to add, store, check and other functions.

    ATTRIBUTES:
        * **x** (*list*): data storage.
        * **mac** (*list*): Store the corresponding message authentication code (MAC).
        * **key** (*list*): Store the corresponding key.
    """

    def __init__(self):
        self.x = []
        self.mac = []
        self.key = []

    def add(self, x, mac, key):
        """
        Adds new data, MAC, and keys to the MacBuffer.

        :param x: data item
        :type x: ReplicatedSecretSharing
        :param mac: message authentication code
        :type mac: ReplicatedSecretSharing
        :param key: secret key
        :type key: ReplicatedSecretSharing
        """
        self.x.append(x.clone())
        self.mac.append(mac.clone())
        self.key.append(key.clone())

    def check(self):
        """
        Check whether the combined data and MAC are consistent to determine whether the calculation result is correct.

        First, merge three lists, and then verify that the combined data and MAC are consistent.After the validation is complete, call *self.__init__()* to reset the instance of this class, clearing out all stored content.
        """
        from NssMPC import ReplicatedSecretSharing
        x = ReplicatedSecretSharing.cat(self.x)
        mac = ReplicatedSecretSharing.cat(self.mac)
        key = ReplicatedSecretSharing.cat(self.key)
        from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import mac_check
        mac_check(x, mac, key)
        self.__init__()


class Register(object):
    """
    The class provides a flexible mechanism to manage module registration and neglect.
    """

    def __init__(self, name):
        """
        ATTRIBUTES:
            * **module_dict** (*dict*): The names of the storage modules and the corresponding module objects.
            * **name** (*str*): the name of the registrar.
            * **ignored_modules** (*set*): Store ignored module names.

        :param name: the name of the registrar.
        :type name: str
        """
        self.module_dict = dict()
        self.name = name
        self.ignored_modules = set()

    def ignore(self):
        """
        Provide a decorator that marks a module as being ignored.

        Define an internal function `_ignore(target)`, add the name of the module to the `self.ignored_modules` set.

        :return: The module is ignored
        :rtype: set
        """

        def _ignore(target):
            self.ignored_modules.add(target.__name__)
            return target

        return _ignore

    def register(self, target):
        """
        Register a module and add it to the module dictionary.

        :param target: The module object to be registered.
        :type target: torch.nn.Module
        :return: Registered module object
        :rtype: torch.nn.Module
        """
        self.module_dict[target.__name__] = target
        return target

    def modules(self):
        """
        Get all unignored modules.

        First, use the dictionary derivation, traverse the items in *self.module_dict*, then, Filter out the modules with names in *self.ignored_modules*.

        :return: A dictionary that only contains modules that have not been ignored.
        :rtype: dict
        """
        return {name: module for name, module in self.module_dict.items() if name not in self.ignored_modules}


MAC_BUFFER = MacBuffer()
ParameterRegistry = Register('ParameterRegistry')
