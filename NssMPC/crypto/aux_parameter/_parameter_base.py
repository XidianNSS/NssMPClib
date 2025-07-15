#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
This module defines classes and methods related to parameters used in the NssMPC framework.
It provides functionality to generate, save, load, and manipulate parameters using pickle serialization.
"""

import os
import pickle
from copy import deepcopy

from NssMPC.config import param_path
from NssMPC.config.runtime import ParameterRegistry


class ParameterMeta(type):
    """
    A metaclass that registers the parameter class in the `ParameterRegistry`.

    This metaclass ensures that any class using `ParameterMeta` as its metaclass
    is automatically registered in the `ParameterRegistry`.

    We mainly use this class to ensure that all classes inherit from Parameter are registered in ParameterRegistry for easy management.
    """

    def __new__(cls, name, bases, dct):
        """
        Creates a new class and registers it in the `ParameterRegistry`.

        :param name: The name of the new class.
        :type name: str
        :param bases: The base classes of the new class.
        :type bases: tuple
        :param dct: The class attributes and methods.
        :type dct: dict
        :return: The newly created class.
        :rtype: type
        """
        ret = super().__new__(cls, name, bases, dct)
        ParameterRegistry.register(ret)
        return ret


@ParameterRegistry.ignore()
class Parameter(metaclass=ParameterMeta):
    """
    A base class for handling parameter-related operations such as generation, saving, loading,
    and manipulation. This class uses the `ParameterMeta` metaclass for automatic registration in the `ParameterRegistry`.
    """

    @staticmethod
    def gen(*args) -> list:
        """
        Generates a list of parameters.

        This method should be implemented in subclasses to generate the parameters.

        :param args: Arguments required for parameter generation.
        :type args: tuple[Any ...]
        :return: A list of generated parameters.
        :rtype: list
        """
        pass

    @classmethod
    def gen_and_save(cls, num, saved_name=None, *args):
        """
        Generates parameters and saves them to the specified path.

        :param num: The number of parameters to generate.
        :type num: int
        :param saved_name: The base name for saving the parameters. If not provided, the class name is used.
        :type saved_name: str, optional
        :param args: Additional arguments for parameter generation.
        :type args: tuple
        """
        file_path = f"{param_path}{cls.__name__}/"
        params = cls.gen(num, *args)
        num_of_party = len(params)
        if saved_name is None:
            saved_name = f"{cls.__name__}"
        for party_id in range(num_of_party):
            name = saved_name + "_" + str(party_id) + ".pth"
            params[party_id].save(file_path, name)

    def save(self, file_path, name=None):
        """
        Saves the current parameter instance to a file.

        The parameter is serialized using `pickle` and saved to the specified path.

        :param file_path: The directory where the parameter will be saved.
        :type file_path: str
        :param name: The name of the file to save. If not provided, defaults to the class name with `.pth` extension.
        :type name: str, optional
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        dic = self.to('cpu').to_dic()
        if name is None:
            name = self.__class__.__name__ + '.pth'
        file_name = os.path.join(file_path, name)
        with open(file_name, 'wb') as file:
            pickle.dump(dic, file)

    @classmethod
    def load(cls, file_path):
        """
        Loads a parameter instance from a file.

        The file is deserialized using `pickle`, and the parameter instance is reconstructed.

        :param file_path: The path to the file from which to load the parameter.
        :type file_path: str
        :return: The loaded parameter instance.
        :rtype: Parameter or dict

        """
        # file_name = os.path.join(file_path, name) `
        with open(file_path, 'rb') as file:
            dic = pickle.load(file)
        param = cls.from_dic(dic)
        return param

    @classmethod
    def from_dic(cls, dic):
        """
        Creates a parameter instance from a dictionary.

        :param dic: A dictionary representation of the parameter.
        :type dic: dict
        :return: A parameter instance with values from the dictionary.
        :rtype: Parameter
        """
        ret = cls()
        for key, value in ret.__dict__.items():
            if hasattr(value, 'from_dic'):
                setattr(ret, key, getattr(ret, key).from_dic(dic[key]))
            else:
                setattr(ret, key, dic[key])
        return ret

    def to_dic(self):
        """
        Converts the parameter instance to a dictionary.

        :return: A dictionary representation of the parameter instance.
        :rtype: dict
        """
        dic = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dic'):
                dic[key] = value.to_dic()
            else:
                dic[key] = value
        return dic

    def __getstate__(self):
        """Returns the state of the parameter instance as a dictionary for serialization."""
        return self.to_dic()

    def __setstate__(self, state):
        """Updates the parameter instance's state from a dictionary."""
        self.__dict__.update(self.from_dic(state).__dict__)

    def __len__(self):
        """Returns the length of the first attribute that has a 'shape' property."""
        fallback = None
        for attr, value in self.__dict__.items():
            if hasattr(value, 'shape'):
                return value.shape[0]
            elif hasattr(value, '__len__'):
                fallback = len(value)
        if fallback is not None:
            return fallback

    def __getitem__(self, item):
        """
        Retrieve parameters (properties) that can be manipulated by the index.

        :param item: The index to access.
        :type item: int or slice
        :return: A new instance of the parameter with the selected attributes.
        :rtype: Parameter
        """
        ret = self.__class__()
        for attr, value in self.__dict__.items():
            if hasattr(value, 'getitem'):
                setattr(ret, attr, value.getitem(item))
            elif hasattr(value, '__getitem__'):
                setattr(ret, attr, value[item])
            else:
                setattr(ret, attr, value)
        return ret

    def __setitem__(self, key, new_value):
        """
        Sets parameters (properties) that can be manipulated by the index.

        :param key: The index to update.
        :type key: int or slice or str
        :param new_value: The new value to set.
        :type new_value: Parameter
        """
        for attr_name, attr_value in self.__dict__.items():
            if hasattr(attr_value, 'setitem'):
                attr_value.setitem(key, getattr(new_value, attr_name, None))
            elif hasattr(attr_value, '__setitem__'):
                attr_value.__setitem__(key, getattr(new_value, attr_name, None))
            else:
                setattr(self, attr_name, getattr(new_value, attr_name, None))

    def clone(self):
        """
        Creates a deep copy of the parameter instance.

        :return: A deep copy of the parameter.
        :rtype: Parameter
        """
        ret = self.__class__()
        for attr, value in self.__dict__.items():
            if hasattr(value, 'clone'):
                setattr(ret, attr, value.clone())
            else:
                setattr(ret, attr, deepcopy(value))
        return ret

    def to(self, device):
        """
        Moves the parameter's attributes to the specified device.

        :param device: The target device ('cpu' or 'gpu').
        :type device: str
        :return: The parameter with attributes moved to the specified device.
        :rtype: Parameter
        """
        for attr, value in self.__dict__.items():
            if hasattr(value, 'to'):
                setattr(self, attr, value.to(device))
            elif isinstance(value, list):
                for i in range(len(value)):
                    if hasattr(value[i], 'to'):
                        value[i] = value[i].to(device)
            elif isinstance(value, dict):
                for k, v in value:
                    if hasattr(v, 'to'):
                        value[k] = v.to(device)
        return self

    def pop(self):
        """
        Removes and returns the last element from the parameter's attributes.

        :return: A new parameter instance with the popped attributes.
        :rtype: Parameter
        """
        ret = self.__class__()
        for attr, value in self.__dict__.items():
            if hasattr(value, 'pop'):
                setattr(ret, attr, value.pop())
            elif hasattr(value, "__getitem__") and not isinstance(value, str):
                setattr(ret, attr, value[-1])
                setattr(self, attr, value[:-1])
            else:
                setattr(ret, attr, value)
        return ret
