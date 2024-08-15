import os
import pickle
from copy import deepcopy

from NssMPC.config import param_path
from NssMPC.config.runtime import ParameterRegistry


class ParameterMeta(type):
    def __new__(cls, name, bases, dct):
        ret = super().__new__(cls, name, bases, dct)
        ParameterRegistry.register(ret)
        return ret


@ParameterRegistry.ignore()
class Parameter(metaclass=ParameterMeta):
    @staticmethod
    def gen(*args) -> list:
        pass

    @classmethod
    def gen_and_save(cls, num, saved_name=None, *args):
        file_path = f"{param_path}{cls.__name__}/"
        params = cls.gen(num, *args)
        num_of_party = len(params)
        if saved_name is None:
            saved_name = f"{cls.__name__}"
        for party_id in range(num_of_party):
            name = saved_name + "_" + str(party_id) + ".pth"
            params[party_id].save(file_path, name)

    def save(self, file_path, name=None):
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
        # file_name = os.path.join(file_path, name) `
        with open(file_path, 'rb') as file:
            dic = pickle.load(file)
        param = cls.from_dic(dic)
        return param

    @classmethod
    def from_dic(cls, dic):
        ret = cls()
        for key, value in ret.__dict__.items():
            if hasattr(value, 'from_dic'):
                setattr(ret, key, getattr(ret, key).from_dic(dic[key]))
            else:
                setattr(ret, key, dic[key])
        return ret

    def to_dic(self):
        dic = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dic'):
                dic[key] = value.to_dic()
            else:
                dic[key] = value
        return dic

    def __getstate__(self):
        return self.to_dic()

    def __setstate__(self, state):
        self.__dict__.update(self.from_dic(state).__dict__)

    def __len__(self):
        for attr, value in self.__dict__.items():
            if hasattr(value, 'shape'):
                return value.shape[0]

    def __getitem__(self, item):
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
        for attr_name, attr_value in self.__dict__.items():
            if hasattr(attr_value, 'setitem'):
                attr_value.setitem(key, getattr(new_value, attr_name, None))
            elif hasattr(attr_value, '__setitem__'):
                attr_value.__setitem__(key, getattr(new_value, attr_name, None))
            else:
                setattr(self, attr_name, getattr(new_value, attr_name, None))

    def clone(self):
        ret = self.__class__()
        for attr, value in self.__dict__.items():
            if hasattr(value, 'clone'):
                setattr(ret, attr, value.clone())
            else:
                setattr(ret, attr, deepcopy(value))
        return ret

    def to(self, device):
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
