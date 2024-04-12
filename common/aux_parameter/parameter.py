import os
import pickle
from copy import deepcopy

from common.tensor.ring_tensor import RingTensor
from config.base_configs import base_path


class Parameter(object):
    @staticmethod
    def gen(*args) -> list:
        pass

    @classmethod
    def gen_and_save(cls, *args):
        params = cls.gen(*args)
        num_of_party = len(params)
        for party_id in range(num_of_party):
            params[party_id].save(party_id, num_of_party)

    def save(self, party_id, num_of_party=2):
        path = f"{base_path}/aux_parameters/{type(self).__name__}/{num_of_party}party/"
        file_name = os.path.join(path, f"{type(self).__name__}_{party_id}.pth")

        if not os.path.exists(path):
            os.makedirs(path)
        dic = self.to('cpu').to_dic()
        with open(file_name, 'wb') as file:
            pickle.dump(dic, file)

    def save_by_name(self, name, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        dic = self.to('cpu').to_dic()
        file_name = os.path.join(file_path, name)
        with open(file_name, 'wb') as file:
            pickle.dump(dic, file)

    @classmethod
    def load(cls, party_id, num_of_party=2):
        path = f"{base_path}/aux_parameters/{cls.__name__}/{num_of_party}party/"
        file_name = os.path.join(path, f"{cls.__name__}_{party_id}.pth")
        with open(file_name, 'rb') as file:
            dic = pickle.load(file)
        param = cls.from_dic(dic)
        return param

    @classmethod
    def load_by_name(cls, name, file_path):
        file_name = os.path.join(file_path, name)
        with open(file_name, 'rb') as file:
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
            if isinstance(value, RingTensor):
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
            if hasattr(attr_value, '__setitem__'):
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
