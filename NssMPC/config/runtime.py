#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

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
        :type x: Any
        :param mac: message authentication code
        :type mac: str
        :param key: secret key
        :type key: str
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
