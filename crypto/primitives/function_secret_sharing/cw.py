from common.tensor.ring_tensor import RingTensor


class CW(object):
    """Correction Words, CW
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, item):
        ret = CW()
        for k, v in self.__dict__.items():
            if hasattr(v, '__getitem__'):
                setattr(ret, k, v[item])
        return ret

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, RingTensor):
                setattr(self, k, v.to(device))
        return self


class CWList(list):
    def __init__(self):
        super().__init__()

    def getitem(self, item):
        ret = CWList()
        for element in self:
            ret.append(element[item])
        return ret

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, CW) or isinstance(v, RingTensor):
                setattr(self, k, v.to(device))
        return self

    def expand_as(self, input):
        ret = CWList()
        for i, value in enumerate(self):
            if hasattr(value, 'expand_as'):
                ret.append(value.expand_as(input[i]))
        return ret
