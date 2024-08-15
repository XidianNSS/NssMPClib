import torch
from NssMPC.common.ring import RingTensor


def auto_delegate(call_methods, delegate_methods):
    def decorator(cls):
        def create_delegated_method(method_name, is_delegate=False):
            if not hasattr(RingTensor, method_name):
                print(f"Warning: RingTensor does not have method '{method_name}'")
                return

            def delegate(self, *args, **kwargs):
                result = getattr(self.item, method_name)(*args, **kwargs)
                if is_delegate:
                    return self.__class__(result, self.party)
                return result

            return delegate

        for name in call_methods:
            setattr(cls, name, create_delegated_method(name))

        for name in delegate_methods:
            setattr(cls, name, create_delegated_method(name, is_delegate=True))

        return cls

    return decorator


@auto_delegate(
    call_methods=['numel', 'tolist'],
    delegate_methods=['__neg__', '__xor__', 'reshape', 'view', 'transpose', 'squeeze', 'unsqueeze', 'flatten', 'clone',
                      'pad', 'sum', 'repeat', 'repeat_interleave', 'permute', 'to', 'contiguous', 'expand'])
class ArithmeticBase:
    def __init__(self, item, party):
        self.item = item
        self.party = party

    @property
    def dtype(self):
        return self.item.dtype

    @dtype.setter
    def dtype(self, value):
        self.item.dtype = value

    @property
    def device(self):
        return self.item.device

    @property
    def shape(self):
        return self.item.shape

    @property
    def scale(self):
        return self.item.scale

    @property
    def bit_len(self):
        return self.item.bit_len

    @bit_len.setter
    def bit_len(self, value):
        self.item.bit_len = value

    def __getstate__(self):
        state = self.__dict__.copy()
        state['party'] = None
        return state

    def __getitem__(self, item):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __matmul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __mod__(self, other):
        raise NotImplementedError

    def __xor__(self, other):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __pow__(self, power, modulo=None):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        raise NotImplementedError

    def __ge__(self, other):
        raise NotImplementedError

    def __le__(self, other):
        raise NotImplementedError

    def __gt__(self, other):
        raise NotImplementedError

    def __lt__(self, other):
        raise NotImplementedError

    @classmethod
    def load_from_file(cls, file_path, party=None):
        result = RingTensor.load_from_file(file_path)
        result.party = party
        return result

    @classmethod
    def cat(cls, dim):
        raise NotImplementedError

    @classmethod
    def stack(cls, dim):
        raise NotImplementedError

    @classmethod
    def roll(cls, shifts, dims):
        raise NotImplementedError

    @classmethod
    def rotate(cls, shifts, dims):
        raise NotImplementedError

    @classmethod
    def max(cls, x, dim=None):
        def max_iterative(inputs):
            while inputs.shape[0] > 1:
                if inputs.shape[0] % 2 == 1:
                    inputs = x.__class__.cat([inputs, inputs[-1:]], 0)
                inputs_0 = inputs[0::2]
                inputs_1 = inputs[1::2]
                ge = inputs_0 >= inputs_1
                inputs = ge * (inputs_0 - inputs_1) + inputs_1
            return inputs

        if dim is None:
            x = x.flatten()
        else:
            x = x.transpose(dim, 0)
        if x.shape[0] == 1:
            result = x.squeeze()
        else:
            result = max_iterative(x)
        if dim is not None:
            result = result.transpose(0, dim)
        return result

    def save(self, path):
        torch.save(self, path)

    def view(self, *shape):
        raise NotImplementedError

    def reshape(self, *shape):
        raise NotImplementedError

    def transpose(self, dim0, dim1):
        raise NotImplementedError

    def permute(self, *dims):
        raise NotImplementedError

    def squeeze(self, dim=None):
        raise NotImplementedError

    def unsqueeze(self, dim):
        raise NotImplementedError

    def flatten(self, start_dim=0, end_dim=-1):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError

    def pad(self, pad, mode='constant', value=0):
        raise NotImplementedError

    def sum(self, dim):
        raise NotImplementedError

    def repeat(self, repeats, dim):
        raise NotImplementedError

    def repeat_interleave(self, repeats, dim):
        raise NotImplementedError

    def to(self, *args, **kwargs):
        raise NotImplementedError

    def size(self, dim=None):
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    def contiguous(self):
        raise NotImplementedError

    def numel(self):
        raise NotImplementedError

    def expand(self, *size):
        raise NotImplementedError


def methods_delegate():
    def decorator(cls):
        def create_delegated_method(method_name):
            def delegate(self, *args, **kwargs):
                result_0 = getattr(self._item_0, method_name)(*args, **kwargs)
                result_1 = getattr(self._item_1, method_name)(*args, **kwargs)
                return self.__class__(result_0, result_1)

            return delegate

        for name in RingTensor.__dict__.keys():
            attr = getattr(RingTensor, name)
            if name not in cls.__dict__ and callable(attr) and not isinstance(attr, (staticmethod, classmethod)):
                setattr(cls, name, create_delegated_method(name))

        return cls

    return decorator


@methods_delegate()
class RingPair:
    def __init__(self, item_0, item_1):
        self._item_0 = item_0
        self._item_1 = item_1

    @property
    def dtype(self):
        return self._item_0.dtype

    @dtype.setter
    def dtype(self, value):
        self._item_0.dtype = value
        self._item_1.dtype = value

    @property
    def device(self):
        return self._item_0.device

    @property
    def shape(self):
        return self._item_0.shape

    @property
    def scale(self):
        return self._item_0.scale

    @property
    def tensor(self):
        return torch.stack((self._item_0.tensor, self._item_1.tensor))

    @property
    def T(self):
        return RingPair(self._item_0.T, self._item_1.T)

    def numel(self):
        return self._item_0.numel()

    def img2col(self, k_size: int, stride: int):
        res0, _, _, _ = self._item_0.img2col(k_size, stride)
        res1, batch, out_size, channel = self._item_1.img2col(k_size, stride)
        return RingPair(res0, res1), batch, out_size, channel

    def __str__(self):
        return f"[value0:{self._item_0}\nvalue1:{self._item_1}]"

    def __getitem__(self, item):
        assert item in [0, 1], IndexError("Index out of range")
        return self._item_0 if item == 0 else self._item_1

    def __setitem__(self, key, value):
        assert key in [0, 1], IndexError("Index out of range")
        if key == 0:
            self._item_0 = value
        else:
            self._item_1 = value
