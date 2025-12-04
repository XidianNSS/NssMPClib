#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

"""
This module defines classes and methods related to parameters used in the NssMPC framework.
It provides functionality to generate, save, load, and manipulate parameters using pickle serialization.
"""
import inspect
import os
import pickle
from copy import deepcopy
from typing import Tuple, Type, Dict, Any

from NssMPC.config import param_path
from NssMPC.infra.mpc.aux_parameter import ParamProvider


class Register(object):
    """The class provides a flexible mechanism to manage module registration and neglect."""

    def __init__(self, name):
        """Initializes the Register.

        Args:
            name (str): The name of the registrar.

        Attributes:
            module_dict (dict): The names of the storage modules and the corresponding module objects.
            name (str): The name of the registrar.
            ignored_modules (set): Store ignored module names.

        Examples:
            >>> registry = Register('ParameterRegistry')
        """
        self.module_dict = dict()
        self.name = name
        self.ignored_modules = set()

    def ignore(self):
        """Provide a decorator that marks a module as being ignored.

        Define an internal function `_ignore(target)`, add the name of the module to the `self.ignored_modules` set.

        Returns:
            function: The decorator function.

        Examples:
            >>> @registry.ignore()
        """

        def _ignore(target):
            self.ignored_modules.add(target.__name__)
            return target

        return _ignore

    def register(self, target):
        """Register a module and add it to the module dictionary.

        Args:
            target (torch.nn.Module): The module object to be registered.

        Returns:
            torch.nn.Module: The registered module object.

        Examples:
            >>> registry.register(MyModule)
        """
        self.module_dict[target.__name__] = target
        return target

    def modules(self):
        """Get all unignored modules.

        First, use the dictionary derivation, traverse the items in *self.module_dict*, then,
        Filter out the modules with names in *self.ignored_modules*.

        Returns:
            dict: A dictionary that only contains modules that have not been ignored.

        Examples:
            >>> mods = registry.modules()
        """
        return {name: module for name, module in self.module_dict.items() if name not in self.ignored_modules}


ParameterRegistry = Register('ParameterRegistry')


class ParamList(list):
    def __init__(self, *args):
        """Initializes a ParamList instance.

        Args:
            *args: Variable length argument list. Can be a single list/tuple or multiple arguments.

        Examples:
            >>> param_list = ParamList([1, 2, 3])
        """
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            super().__init__(self._convert_item(args[0]))
        else:
            super().__init__(self._convert_item(args))

    def _convert_item(self, item):
        """Recursively converts nested structures.

        Args:
            item: The item to convert.

        Returns:
            The converted item, potentially a ParamList or ParamDict.

        Examples:
            >>> converted = param_list._convert_item([1, 2])
        """
        if isinstance(item, (list, tuple)):
            return [self._convert_item(x) for x in item]
        elif isinstance(item, dict):
            return ParamDict(item)
        return item

    # def __setitem__(self, index, value):
    #     super().__setitem__(index, self._convert_item(value))

    def append(self, value):
        """Appends an item to the end of the list, converting it if necessary.

        Args:
            value: The item to append.

        Examples:
            >>> param_list.append({'a': 1})
        """
        super().append(self._convert_item(value))

    def extend(self, values):
        """Extends the list by appending elements from the iterable, converting them if necessary.

        Args:
            values: The iterable elements to extend.

        Examples:
            >>> param_list.extend([4, 5])
        """
        super().extend(self._convert_item(values))

    def to(self, device):
        """Moves all elements to the specified device.

        Args:
            device: The target device (e.g., 'cpu', 'cuda').

        Returns:
            ParamList: A new ParamList with elements moved to the device.

        Examples:
            >>> new_list = param_list.to('cuda')
        """
        result = ParamList()
        for v in self:
            if hasattr(v, 'to') and callable(getattr(v, 'to')):
                result.append(v.to(device))
            else:
                result.append(v)
        return result


class ParamDict(dict):
    def __init__(self, *args, **kwargs):
        """Initializes a ParamDict instance.

        Supports initialization via a dictionary or keyword arguments.

        Args:
            *args: Positional arguments. Expected to be a single dictionary if provided.
            **kwargs: Keyword arguments.

        Raises:
            TypeError: If more than one positional argument is provided or if the argument is not a dictionary.

        Examples:
            >>> param_dict = ParamDict({'a': 1})
            >>> param_dict = ParamDict(a=1)
        """
        super().__init__()
        if args:
            if len(args) > 1:
                raise TypeError("ParamDict expected at most 1 argument, got {}".format(len(args)))
            if isinstance(args[0], dict):
                for k, v in args[0].items():
                    self[k] = self._convert_value(v)
            else:
                raise TypeError("ParamDict argument must be a dictionary")

        # 处理关键字参数
        for k, v in kwargs.items():
            self[k] = self._convert_value(v)

    def _convert_value(self, value):
        """Recursively converts nested lists, tuples, and dictionaries to ParamList and ParamDict.

        Args:
            value: The value to convert.

        Returns:
            The converted value.

        Examples:
            >>> converted = param_dict._convert_value({'b': 2})
        """
        if isinstance(value, (list, tuple)):
            return ParamList(value)
        elif isinstance(value, dict):
            return ParamDict(value)
        else:
            return value

    # def __setitem__(self, key, value):
    #     """设置项，自动转换嵌套结构"""
    #     super().__setitem__(key, self._convert_value(value))

    def update(self, other=None, **kwargs):
        """Updates the dictionary with elements from another dictionary or iterable of key/value pairs.

        Args:
            other: The dictionary or iterable to update from.
            **kwargs: Keyword arguments to update from.

        Examples:
            >>> param_dict.update({'c': 3})
        """
        if other is not None:
            if hasattr(other, 'keys'):
                for key in other:
                    self[key] = other[key]
            else:
                for key, value in other:
                    self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def to(self, device):
        """Moves all supported elements to the specified device.

        Args:
            device: The target device.

        Returns:
            ParamDict: A new ParamDict with elements moved to the device.

        Examples:
            >>> new_dict = param_dict.to('cuda')
        """
        result = ParamDict()
        for k, v in self.items():
            if hasattr(v, 'to') and callable(getattr(v, 'to')):
                result[k] = v.to(device)
            elif isinstance(v, (ParamList, ParamDict)):
                result[k] = v.to(device)
            else:
                result[k] = v
        return result

    def copy(self):
        """Creates a deep copy of the dictionary.

        Returns:
            ParamDict: A deep copy of the dictionary.

        Examples:
            >>> dict_copy = param_dict.copy()
        """
        return ParamDict(super().copy())


class ParameterMeta(type):

    def __new__(cls, name: str, bases: Tuple[Type, ...], attrs: Dict[str, Any]) -> Type:
        """Creates a new class, converting list/dict attributes to ParamList/ParamDict and registering the class.

        Args:
            name: The name of the class.
            bases: The base classes.
            attrs: The class attributes.

        Returns:
            Type: The created class.

        Examples:
            >>> class MyParam(metaclass=ParameterMeta): pass
        """
        # 转换类属性
        for attr_name, attr_value in list(attrs.items()):
            if not attr_name.startswith('__') and not inspect.ismethod(attr_value) and not inspect.isfunction(
                    attr_value):
                if type(attr_value) in (list, tuple):
                    attrs[attr_name] = ParamList(attr_value)
                elif type(attr_value) is dict:
                    attrs[attr_name] = ParamDict(attr_value)

        ret = super().__new__(cls, name, bases, attrs)
        ParameterRegistry.register(ret)
        return ret

    def __call__(cls, *args, **kwargs) -> Any:
        """Creates an instance of the class, converting instance attributes to ParamList/ParamDict.

        Args:
            *args: Positional arguments for the constructor.
            **kwargs: Keyword arguments for the constructor.

        Returns:
            Any: The created instance.

        Examples:
            >>> instance = MyParam()
        """
        instance = super().__call__(*args, **kwargs)

        for attr_name in dir(instance):
            if not attr_name.startswith('__'):
                try:
                    attr_value = getattr(instance, attr_name)
                    if type(attr_value) in (list, tuple):
                        setattr(instance, attr_name, ParamList(attr_value))
                    elif type(attr_value) is dict:
                        setattr(instance, attr_name, ParamDict(attr_value))
                except (AttributeError, TypeError):
                    continue

        return instance


@ParameterRegistry.ignore()
class Parameter(metaclass=ParameterMeta):
    """
    A base class for handling parameter-related operations such as generation, saving, loading,
    and manipulation. This class uses the `ParameterMeta` metaclass for automatic registration in the `ParameterRegistry`.
    """

    @staticmethod
    def gen(*args) -> list:
        """Generates a list of parameters.

        This method should be implemented in subclasses to generate the parameters.

        Args:
            *args: Arguments required for parameter generation.

        Returns:
            list: A list of generated parameters.

        Examples:
            >>> params = Parameter.gen(10)
        """
        pass

    @classmethod
    def gen_and_save(cls, num, saved_name=None, *args):
        """Generates parameters and saves them to the specified path.

        Args:
            num (int): The number of parameters to generate.
            saved_name (str, optional): The base name for saving the parameters. If not provided, the class name is used.
            *args: Additional arguments for parameter generation.

        Examples:
            >>> Parameter.gen_and_save(10, 'test_param')
        """
        file_path = f"{param_path}{cls.__name__}/"
        params = cls.gen(num, *args)
        num_of_party = len(params)
        if saved_name is None:
            saved_name = f"{cls.__name__}"
        for party_id in range(num_of_party):
            name = saved_name + "_" + str(party_id) + ".pth"
            params[party_id].save(file_path, name)

    @classmethod
    def load_provider(cls, party):
        """Loads a parameter provider for the specified party.

        Args:
            party: The party instance requiring the provider.

        Returns:
            ParamProvider: The loaded parameter provider.

        Examples:
            >>> provider = Parameter.load_provider(party)
        """
        provider = ParamProvider(param_type=cls)
        provider.load_param(provider.saved_name + '_' + str(party.party_id) + '.pth')
        party.append_provider(provider)
        return provider

    def save(self, file_path, name=None):
        """Saves the current parameter instance to a file.

        The parameter is serialized using `pickle` and saved to the specified path.

        Args:
            file_path (str): The directory where the parameter will be saved.
            name (str, optional): The name of the file to save. If not provided, defaults to the class name with `.pth` extension.

        Examples:
            >>> param.save('./params', 'my_param.pth')
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
        """Loads a parameter instance from a file.

        The file is deserialized using `pickle`, and the parameter instance is reconstructed.

        Args:
            file_path (str): The path to the file from which to load the parameter.

        Returns:
            Parameter: The loaded parameter instance.

        Examples:
            >>> param = Parameter.load('./params/my_param.pth')
        """
        # file_name = os.path.join(file_path, name) `
        with open(file_path, 'rb') as file:
            dic = pickle.load(file)
        param = cls.from_dic(dic)
        return param

    @classmethod
    def from_dic(cls, dic):
        """Creates a parameter instance from a dictionary.

        Args:
            dic (dict): A dictionary representation of the parameter.

        Returns:
            Parameter: A parameter instance with values from the dictionary.

        Examples:
            >>> param = Parameter.from_dic(param_dict)
        """
        ret = cls()
        for key, value in ret.__dict__.items():
            if hasattr(value, 'from_dic'):
                setattr(ret, key, getattr(ret, key).from_dic(dic[key]))
            else:
                setattr(ret, key, dic[key])
        return ret

    def to_dic(self):
        """Converts the parameter instance to a dictionary.

        Returns:
            dict: A dictionary representation of the parameter instance.

        Examples:
            >>> dic = param.to_dic()
        """
        dic = ParamDict()
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dic'):
                dic[key] = value.to_dic()
            else:
                dic[key] = value
        return dic

    def __getstate__(self):
        """Returns the state of the parameter instance as a dictionary for serialization.

        Returns:
            dict: The state dictionary.

        Examples:
            >>> state = param.__getstate__()
        """
        return self.to_dic()

    def __setstate__(self, state):
        """Updates the parameter instance's state from a dictionary.

        Args:
            state (dict): The state dictionary.

        Examples:
            >>> param.__setstate__(state_dict)
        """
        self.__dict__.update(self.from_dic(state).__dict__)

    def __len__(self):
        """Returns the length of the first attribute that has a 'shape' property or is a list.

        Returns:
            int: The length of the parameter.

        Examples:
            >>> length = len(param)
        """
        fallback = None
        for attr, value in self.__dict__.items():
            if hasattr(value, 'shape'):
                return value.shape[0]
            elif hasattr(value, '__len__'):
                fallback = len(value)
        if fallback is not None:
            return fallback

    def __getitem__(self, item):
        """Retrieve parameters (properties) that can be manipulated by the index.

        Args:
            item (int | slice): The index or slice to access.

        Returns:
            Parameter: A new instance of the parameter with the selected attributes.

        Examples:
            >>> sub_param = param[0]
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
        """Sets parameters (properties) that can be manipulated by the index.

        Args:
            key (int | slice | str): The index to update.
            new_value (Parameter): The new value to set.

        Examples:
            >>> param[0] = new_param
        """
        for attr_name, attr_value in self.__dict__.items():
            if hasattr(attr_value, 'setitem'):
                attr_value.setitem(key, getattr(new_value, attr_name, None))
            elif hasattr(attr_value, '__setitem__'):
                attr_value.__setitem__(key, getattr(new_value, attr_name, None))
            else:
                setattr(self, attr_name, getattr(new_value, attr_name, None))

    def clone(self):
        """Creates a deep copy of the parameter instance.

        Returns:
            Parameter: A deep copy of the parameter.

        Examples:
            >>> param_copy = param.clone()
        """
        ret = self.__class__()
        for attr, value in self.__dict__.items():
            if hasattr(value, 'clone'):
                setattr(ret, attr, value.clone())
            else:
                setattr(ret, attr, deepcopy(value))
        return ret

    def to(self, device):
        """Moves the parameter's attributes to the specified device.

        Args:
            device (str): The target device ('cpu' or 'gpu').

        Returns:
            Parameter: The parameter with attributes moved to the specified device.

        Examples:
            >>> param = param.to('cuda')
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
        """Removes and returns the last element from the parameter's attributes.

        Returns:
            Parameter: A new parameter instance with the popped attributes.

        Examples:
            >>> last_param = param.pop()
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
