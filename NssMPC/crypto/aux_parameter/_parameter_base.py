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
from NssMPC.config.runtime import ParameterRegistry

class ParamList(list):
    def __init__(self, *args):
        """初始化 ParamList 实例"""
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            super().__init__(self._convert_item(args[0]))
        else:
            super().__init__(self._convert_item(args))

    def _convert_item(self, item):
        """递归转换嵌套结构"""
        if isinstance(item, (list, tuple)):
            return [self._convert_item(x) for x in item]
        elif isinstance(item, dict):
            return ParamDict(item)
        return item

    # def __setitem__(self, index, value):
    #     super().__setitem__(index, self._convert_item(value))

    def append(self, value):
        super().append(self._convert_item(value))

    def extend(self, values):
        super().extend(self._convert_item(values))

    def to(self, device):
        result = ParamList()
        for v in self:
            if hasattr(v, 'to') and callable(getattr(v, 'to')):
                result.append(v.to(device))
            else:
                result.append(v)
        return result

class ParamDict(dict):
    def __init__(self, *args, **kwargs):
        """
        初始化 ParamDict 实例
        支持 ParamDict({'a': 1, 'b': 2}) 和 ParamDict(a=1, b=2) 两种方式
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
        """递归转换嵌套的列表、元组和字典为 ParamList 和 ParamDict"""
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
        """更新字典，自动转换嵌套结构"""
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
        """将所有支持 .to() 方法的元素转移到指定设备"""
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
        """创建深拷贝"""
        return ParamDict(super().copy())


class ParameterMeta(type):

    def __new__(cls, name: str, bases: Tuple[Type, ...], attrs: Dict[str, Any]) -> Type:
        # 转换类属性
        for attr_name, attr_value in list(attrs.items()):
            if not attr_name.startswith('__') and not inspect.ismethod(attr_value) and not inspect.isfunction(attr_value):
                if type(attr_value) in (list, tuple):
                    attrs[attr_name] = ParamList(attr_value)
                elif type(attr_value) is dict:
                    attrs[attr_name] = ParamDict(attr_value)

        ret = super().__new__(cls, name, bases, attrs)
        ParameterRegistry.register(ret)
        return ret

    def __call__(cls, *args, **kwargs) -> Any:
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
        dic = ParamDict()
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
