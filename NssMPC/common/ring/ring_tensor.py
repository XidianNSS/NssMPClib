import math
from functools import singledispatchmethod
import torch
import torch.nn.functional as F

from NssMPC.common.utils import cuda_matmul
from NssMPC.config import BIT_LEN, DEVICE, data_type, DTYPE_MAPPING, DTYPE_SCALE_MAPPING, HALF_RING


class RingTensor(object):
    # TODO 运算符重载在不支持类型时应 return NotImplemented 否则不会去检查右值是否支持了此运算
    # TODO 支持可变bit_len?
    """
    Define the tensor on the ring
    Support for the part of operations on pytorch tensor

    Attributes:
        tensor: the pytorch tensor
        dtype: the dtype of the tensor
        bit_len: the expected length of the binary representation after conversion
    Properties:
        scale: the scale of the RingTensor
        device: the device of the tensor
        shape: the shape of the tensor
    """

    @singledispatchmethod
    def __init__(self):
        self.bit_len = None
        self.dtype = None
        self.tensor = None

    @__init__.register(torch.Tensor)
    def from_tensor(self, tensor, dtype='int', device=DEVICE, bit_len=BIT_LEN):
        """Encapsulate a torch.Tensor as a RingTensor.
        """
        self.tensor = tensor.to(device) if tensor.device.type != device else tensor
        self.dtype = dtype
        self.bit_len = bit_len

    @__init__.register(int)
    @__init__.register(list)
    def from_item(self, item, dtype='int', device=DEVICE, bit_len=BIT_LEN):
        """
        Encapsulate an int or list item as a RingTensor.
        """
        self.tensor = torch.tensor(item, dtype=data_type, device=device)
        self.dtype = dtype
        self.bit_len = bit_len

    @property
    def scale(self):
        """
        Return the scale of the RingTensor
        """
        return DTYPE_SCALE_MAPPING[self.dtype]

    @property
    def device(self):
        return self.tensor.device.__str__()

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def T(self):
        new_value = self.tensor.T
        return self.__class__(new_value, self.dtype, self.device, self.bit_len)

    def __str__(self):
        return f"{self.__class__.__name__}\n value:{self.tensor} \n dtype:{self.dtype} \n scale:{self.scale}"

    def __getitem__(self, item):
        return self.__class__(self.tensor[item], self.dtype, self.device, self.bit_len)

    def __setitem__(self, key, value):
        if isinstance(value, RingTensor):
            self.tensor[key] = value.tensor.clone()
        else:
            raise TypeError(f"unsupported operand type(s) for setitem '{type(self)}' and {type(value)}")

    def __len__(self):
        return len(self.tensor)

    def __invert__(self):
        return self.__class__(~self.tensor, self.dtype, self.device, self.bit_len)

    def __add__(self, other):
        if isinstance(other, RingTensor):
            new_value = self.tensor + other.tensor
            bit_len = max(self.bit_len, other.bit_len)
        elif isinstance(other, torch.Tensor) or isinstance(other, int):  # TODO int转型
            new_value = self.tensor + other
            bit_len = self.bit_len
        else:
            return NotImplemented
        return self.__class__(new_value, self.dtype, self.device, bit_len)

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, RingTensor):
            self.tensor += other.tensor
            self.bit_len = max(self.bit_len, other.bit_len)
        elif isinstance(other, torch.Tensor) or isinstance(other, int):
            self.tensor += other
        else:
            raise TypeError("unsupported operand type(s) for += 'RingTensor' and ", type(other))
        return self

    def __sub__(self, other):
        if isinstance(other, RingTensor):
            new_value = self.tensor - other.tensor
            bit_len = max(self.bit_len, other.bit_len)
        elif isinstance(other, (int, torch.Tensor)):
            new_value = self.tensor - other
            bit_len = self.bit_len
        else:
            return NotImplemented
        return self.__class__(new_value, self.dtype, self.device, bit_len)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __isub__(self, other):
        if isinstance(other, RingTensor):
            self.tensor -= other.tensor
            self.bit_len = max(self.bit_len, other.bit_len)
        elif isinstance(other, torch.Tensor) or isinstance(other, int):
            self.tensor -= other
        else:
            raise TypeError("unsupported operand type(s) for -= 'RingTensor' and ", type(other))
        return self

    def __mul__(self, other):
        if isinstance(other, RingTensor):
            scale = min(self.scale, other.scale)
            dtype = self.dtype if self.scale >= other.scale else other.dtype
            new_value = (self.tensor * other.tensor) // scale
        elif isinstance(other, int):
            dtype = self.dtype
            new_value = self.tensor * other
        else:
            return NotImplemented
        return self.__class__(new_value, dtype, self.device)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, RingTensor):
            self.tensor *= other.tensor
            self.tensor //= min(self.scale, other.scale)
            self.dtype = self.dtype if self.scale >= other.scale else other.dtype
        elif isinstance(other, int):
            self.tensor *= other
        else:
            raise TypeError("unsupported operand type(s) for *= 'RingTensor' and ", type(other))
        return self

    def __mod__(self, other):
        if isinstance(other, int):
            new_value = (self.tensor % other)
            bit_len = math.ceil(math.log2(other)) if self.bit_len > math.log2(other) else self.bit_len
        else:
            raise TypeError(
                "unsupported operand type(s) for % 'RingTensor' and ", type(other), 'only support int type')
        return self.__class__(new_value, self.dtype, self.device, bit_len)

    def __matmul__(self, other):
        if isinstance(other, RingTensor):
            assert self.dtype == other.dtype, "dtype not equal"
            if self.device in ('cuda', 'cuda:0', 'cuda:1'):
                new_value = cuda_matmul(self.tensor, other.tensor) // self.scale
            else:
                new_value = torch.matmul(self.tensor, other.tensor) // self.scale
        else:
            raise TypeError(
                "unsupported operand type(s) for @ 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    def __truediv__(self, other):
        if isinstance(other, RingTensor):
            assert self.scale == other.scale, "data must have the same scale"
            new_value = self.tensor / other.tensor * self.scale
        elif isinstance(other, torch.Tensor):
            new_value = self.tensor / other
        elif isinstance(other, int):
            new_value = self.tensor / other
        else:
            raise TypeError("unsupported operand type(s) for / 'RingTensor' and ", type(other))
        return self.__class__(torch.round(new_value), self.dtype, self.device)

    def __itruediv__(self, other):
        if isinstance(other, RingTensor):
            assert self.scale == other.scale, "data must have the same scale"
            self.tensor /= other.tensor
            self.tensor *= self.scale
        elif isinstance(other, torch.Tensor):
            self.tensor /= other
        elif isinstance(other, int):
            self.tensor /= other
        else:
            raise TypeError("unsupported operand type(s) for / 'RingTensor' and ", type(other))
        return self

    def __floordiv__(self, other):
        if isinstance(other, RingTensor):
            assert self.scale == other.scale, "data must have the same scale"
            new_value = self.tensor // other.tensor * self.scale
        elif isinstance(other, torch.Tensor):
            new_value = self.tensor // other
        elif isinstance(other, int):
            new_value = self.tensor // other
        else:
            raise TypeError("unsupported operand type(s) for // 'RingTensor' and ", type(other))
        return self.__class__(new_value, self.dtype, self.device)

    def __ifloordiv__(self, other):
        if isinstance(other, RingTensor):
            assert self.scale == other.scale, "data must have the same scale"
            self.tensor //= other.tensor
            self.tensor *= self.scale
        elif isinstance(other, torch.Tensor):
            self.tensor //= other
        elif isinstance(other, int):
            self.tensor //= other
        else:
            raise TypeError("unsupported operand type(s) for // 'RingTensor' and ", type(other))
        return self

    # neg function on ring (-plaintext)
    def __neg__(self):
        new_value = -self.tensor
        return self.__class__(new_value, dtype=self.dtype, device=self.device)

    def __eq__(self, other):
        if isinstance(other, RingTensor):
            new_value = (self.tensor == other.tensor)
        elif isinstance(other, int):
            new_value = (self.tensor == other * self.scale)
        else:
            raise TypeError(
                "unsupported operand type(s) for == 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    def __ne__(self, other):
        if isinstance(other, RingTensor):
            new_value = self.tensor != other.tensor
        elif isinstance(other, int):
            new_value = self.tensor != other * self.scale
        else:
            raise TypeError(
                "unsupported operand type(s) for != 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    def __gt__(self, other):
        if isinstance(other, RingTensor):
            new_value = (self.tensor > other.tensor)
        elif isinstance(other, int):
            new_value = (self.tensor > other * self.scale)
        else:
            raise TypeError(
                "unsupported operand type(s) for > 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    def __ge__(self, other):
        if isinstance(other, RingTensor):
            new_value = (self.tensor >= other.tensor)
        elif isinstance(other, int):
            new_value = (self.tensor >= other * self.scale)
        else:
            raise TypeError(
                "unsupported operand type(s) for >= 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    def __lt__(self, other):
        if isinstance(other, RingTensor):
            new_value = (self.tensor < other.tensor)
        elif isinstance(other, int):
            new_value = (self.tensor < other * self.scale)
        else:
            raise TypeError(
                "unsupported operand type(s) for < 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    def __le__(self, other):
        if isinstance(other, RingTensor):
            new_value = (self.tensor <= other.tensor)
        elif isinstance(other, int):
            new_value = (self.tensor <= other * self.scale)
        else:
            raise TypeError(
                "unsupported operand type(s) for <= 'RingTensor' and ", type(other), 'please convert to '
                                                                                     'ring first')
        return self.__class__(new_value, self.dtype, self.device)

    def __xor__(self, other):
        if isinstance(other, RingTensor):
            return self.__class__(self.tensor ^ other.tensor, self.dtype, self.device)
        elif isinstance(other, (int, torch.Tensor)):
            return self.__class__(self.tensor ^ other, self.dtype, self.device)
        else:
            raise TypeError(
                "unsupported operand type(s) for ^ 'RingTensor' and ", type(other), 'please convert to ring first')

    def __or__(self, other):
        if isinstance(other, RingTensor):
            return self.__class__(self.tensor | other.tensor, self.dtype, self.device)
        elif isinstance(other, int):
            return self.__class__(self.tensor | other, self.dtype, self.device)
        else:
            raise TypeError(
                "unsupported operand type(s) for | 'RingTensor' and ", type(other), 'please convert to ring first')

    def __and__(self, other):
        if isinstance(other, RingTensor):
            return self.__class__(self.tensor & other.tensor, self.dtype, self.device)
        elif isinstance(other, int):
            return self.__class__(self.tensor & other, self.dtype, self.device)
        else:
            raise TypeError(
                "unsupported operand type(s) for & 'RingTensor' and ", type(other), 'please convert to ring first')

    def __rshift__(self, other):
        if isinstance(other, RingTensor):
            return self.__class__(self.tensor >> other.tensor, self.dtype, self.device)
        elif isinstance(other, int):
            return self.__class__(self.tensor >> other, self.dtype, self.device)

        else:
            raise TypeError(
                "unsupported operand type(s) for >> 'RingTensor' and ", type(other), 'please convert to ring first')

    def __lshift__(self, other):
        if isinstance(other, RingTensor):
            return self.__class__(self.tensor << other.tensor, self.dtype, self.device)
        elif isinstance(other, int):
            return self.__class__(self.tensor << other, self.dtype)
        else:
            raise TypeError(
                "unsupported operand type(s) for << 'RingTensor' and ", type(other), 'please convert to ring first')

    @classmethod
    def convert_to_ring(cls, item):
        """
        Converts the item to RingTensor

        Args:
            item: item to convert

        Returns:
            RingTensor: A RingTensor object
        """
        if isinstance(item, (int, float, list)):
            item = torch.tensor(item, device=DEVICE)
        assert isinstance(item, torch.Tensor), f"unsupported data type(s): {type(item)}"
        scale = DTYPE_MAPPING[item.dtype]
        v = torch.round(item * scale) if scale != 1 else item
        dtype = 'int' if scale == 1 else 'float'
        r = cls(v.to(data_type), dtype=dtype, device=DEVICE)
        return r

    @classmethod
    def load_from_file(cls, file_path):
        """
        Loads a RingTensor from a file
        """
        return torch.load(file_path)

    @classmethod
    def mul(cls, x, y):
        """
        Multiplies two RingTensors without truncation

        Args:
            x (RingTensor)
            y (RingTensor)

        Returns:
            RingTensor: A RingTensor object
        """
        return cls(x.tensor * y.tensor, x.dtype)

    @classmethod
    def matmul(cls, x, y):
        """
        Matrix Multiplies two RingTensors without truncation

        Args:
            x (RingTensor)
            y (RingTensor)

        Returns:
            RingTensor: A RingTensor object
        """
        if x.device != y.device:
            raise TypeError(
                "Expected all ring tensors to be on the same device, but found at least two devices,"
                + f" {x.device} and {y.device}!")
        if x.device == 'cpu':
            return cls(torch.matmul(x.tensor, y.tensor), x.dtype)
        if x.device in ('cuda', 'cuda:0', 'cuda:1'):
            return cls(cuda_matmul(x.tensor, y.tensor), x.dtype)

    @classmethod
    def cat(cls, tensor_list, dim=0):
        return cls(torch.cat([t.tensor for t in tensor_list], dim), tensor_list[0].dtype, tensor_list[0].device,
                   tensor_list[0].bit_len)
        # assert all([t.dtype == tensor_list[0].dtype for t in tensor_list]), "The element type should be the same"
        # if isinstance(tensor_list[0], RingTensor):
        #     def fast_concat(tensors, dim=0):
        #         shape = list(tensors[0].shape)
        #         shape[dim] = sum([t.shape[dim] for t in tensors])
        #         result = torch.empty(*shape, dtype=tensors[0].dtype, device=tensors[0].device)
        #         offset = 0
        #         for t in tensors:
        #             size = t.shape[dim]
        #             result.narrow(dim, offset, size).copy_(t)
        #             offset += size
        #         return result
        #
        #     return cls(fast_concat([t.tensor for t in tensor_list], dim), tensor_list[0].dtype)
        # else:
        #     raise TypeError(f"unsupported operand type(s) for cat '{cls.__name__}' and {type(tensor_list[0])}")

    @classmethod
    def stack(cls, tensor_list, dim=0):
        assert all([t.dtype == tensor_list[0].dtype for t in tensor_list]), "The element type should be the same"
        return cls(torch.stack([t.tensor for t in tensor_list], dim), tensor_list[0].dtype, tensor_list[0].device,
                   tensor_list[0].bit_len)

    @classmethod
    def gather(cls, input, dim, index):
        if isinstance(index, RingTensor):
            index = index.tensor
        return cls(torch.gather(input.tensor, dim, index), input.dtype, input.device, input.bit_len)

    @classmethod
    def roll(cls, input, shifts, dims=0):
        return cls(torch.roll(input.tensor, shifts=shifts, dims=dims), input.dtype, input.device, input.bit_len)

    @classmethod
    def rotate(cls, input, shifts):
        """
        For the input 2-dimensional RingTensor, rotate each row to the left or right
        if shifts > 0, rotate to the right
        if shifts < 0, rotate to the left
        """
        if isinstance(shifts, list):
            shifts = torch.tensor(shifts, dtype=data_type, device=input.device)
        elif isinstance(shifts, RingTensor):
            shifts = shifts.tensor

        n = input.shape[1]
        rows = torch.arange(input.shape[0]).view(-1, 1)
        indices = (torch.arange(n, device=shifts.device) - shifts.view(-1, 1)) % n
        result = input[rows, indices]

        return result

    @classmethod
    def onehot(cls, input, num_classes=-1):
        """
        Creates a one-hot RingTensor from input RingTensor

        Args:
            input (RingTensor): the indices
            num_classes (int): the number of classes

        Returns:
            RingTensor: A one-hot tensor
        """
        return cls(F.one_hot(input.tensor.to(torch.int64), num_classes).to(data_type) * input.scale, input.dtype,
                   input.device)

    @classmethod
    def exp(cls, x):
        assert x.dtype == 'float', "The input tensor should be float type"
        return cls.convert_to_ring(torch.exp(x.convert_to_real_field()))

    @classmethod
    def exp2(cls, x):
        return cls.convert_to_ring(torch.exp2(x.convert_to_real_field()))

    @classmethod
    def tanh(cls, x):
        return cls.convert_to_ring(torch.tanh(x.convert_to_real_field()))

    def convert_to_real_field(self):
        """
        convert the tensor from ring field to real field
        """
        torch_type = {64: {'int': torch.int64, 'float': torch.float64},
                      32: {'int': torch.int32, 'float': torch.float32}}
        return (self.tensor / self.scale).to(torch_type[BIT_LEN][self.dtype])

    def sum(self, dim=0):
        new_value = torch.sum(self.tensor, dim=dim, dtype=self.tensor.dtype)
        return self.__class__(new_value, dtype=self.dtype, device=self.device, bit_len=self.bit_len)

    def all(self):
        return self.tensor.all()

    def any(self):
        return self.tensor.any()

    def to(self, device):
        self.tensor = self.tensor.to(device)
        return self

    def cpu(self):
        self.tensor = self.tensor.cpu()
        return self

    def cuda(self, device=None):
        self.tensor = self.tensor.cuda(device)
        return self

    def save(self, file_path):
        torch.save(self, file_path)

    def clone(self):
        clone = self.__class__(self.tensor.clone(), dtype=self.dtype, device=self.device)
        clone.bit_len = self.bit_len
        return clone

    def get_bit(self, item):
        """
        get a bit from the RingTensor
        Args:
            item:

        Returns: RingTensor

        """
        assert (self.bit_len >> item >= 0), "bit index out of range"
        return RingTensor((self.tensor >> item) & 1)

    def get_tensor_bit(self, item):
        """
        get a bit from the RingTensor
        Args:
            item:

        Returns: torch.Tensor

        """
        assert (self.bit_len >> item >= 0), "bit index out of range"
        return (self.tensor >> item) & 1

    def reshape(self, *shape):
        new = self.__class__(self.tensor.reshape(*shape), dtype=self.dtype, device=self.device)
        new.bit_len = self.bit_len
        return new

    def img2col(self, k_size: int, stride: int):
        """
        Img2Col algorithm using PyTorch's unfold to speed up convolution and pooling

        Args:
            k_size: the size of the kernel
            stride: the stride of the convolution and pooling

        Returns:
            col: the RingTensor object
            batch
            out_size: the size of the col
            channel
        """
        img = self.tensor
        batch, channel, height, width = img.shape
        out_h = (height - k_size) // stride + 1
        out_w = (width - k_size) // stride + 1
        out_size = out_h * out_w
        unfolded = img.unfold(2, k_size, stride).unfold(3, k_size, stride)
        col = unfolded.permute(0, 1, 4, 5, 2, 3)
        col = col.contiguous().view(batch, channel, k_size * k_size, out_h * out_w)
        col = self.__class__(col, dtype=self.dtype, device=self.device)

        return col, batch, out_size, channel

    def repeat_interleave(self, repeats, dim):
        return self.__class__(self.tensor.repeat_interleave(repeats, dim), dtype=self.dtype, device=self.device)

    def repeat(self, *sizes):
        return self.__class__(self.tensor.repeat(*sizes), dtype=self.dtype, device=self.device)

    def transpose(self, dim0, dim1):
        return self.__class__(self.tensor.transpose(dim0, dim1), dtype=self.dtype, device=self.device)

    def pad(self, pad, mode='constant', value=0):
        return self.__class__(F.pad(self.tensor, pad, mode, value), dtype=self.dtype, device=self.device)

    def squeeze(self, dim=-1):
        return self.__class__(self.tensor.squeeze(dim), dtype=self.dtype, device=self.device)

    def unsqueeze(self, dim=-1):
        return self.__class__(self.tensor.unsqueeze(dim), dtype=self.dtype, device=self.device)

    def size(self, dim=None):
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    def view(self, *args):
        view = self.__class__(self.tensor.view(*args), dtype=self.dtype, device=self.device)
        view.bit_len = self.bit_len
        return view

    def flatten(self, start_dim=0, end_dim=-1):
        new = self.__class__(self.tensor.flatten(start_dim, end_dim), dtype=self.dtype, device=self.device)
        new.bit_len = self.bit_len
        return new

    def permute(self, *dims):
        new = self.__class__(self.tensor.permute(dims=dims), dtype=self.dtype, device=self.device)
        new.bit_len = self.bit_len
        return new

    def contiguous(self):
        new = self.__class__(self.tensor.contiguous(), dtype=self.dtype, device=self.device)
        new.bit_len = self.bit_len
        return new

    def tolist(self):
        return self.tensor.tolist()

    def numel(self):
        return self.tensor.numel()

    def signbit(self):
        msb = torch.signbit(self.tensor) + 0
        return self.__class__(msb.to(data_type), self.dtype)

    def bit_slice(self, start, end):
        assert (self.bit_len >> start >= 0), "bit index out of range"
        assert (self.bit_len >> end >= 0), "bit index out of range"

        if start == 0 and end == self.bit_len:
            return self

        if end == self.bit_len:
            return self >> start

        shift_right = self >> start
        new_end = end - start
        mask = (1 << (new_end + 1)) - 1
        masked_value = shift_right & mask
        return masked_value

    def expand(self, *sizes):
        return self.__class__(self.tensor.expand(*sizes), dtype=self.dtype, device=self.device)

    @staticmethod
    def where(condition, x, y):
        if isinstance(x, RingTensor) and isinstance(y, RingTensor):
            return RingTensor(torch.where(condition.tensor.bool(), x.tensor, y.tensor), x.dtype, x.device)
        elif isinstance(x, RingTensor) and isinstance(y, int):
            return RingTensor(torch.where(condition.tensor.bool(), x.tensor, y), x.dtype, x.device)
        elif isinstance(x, int) and isinstance(y, RingTensor):
            return RingTensor(torch.where(condition.tensor.bool(), x, y.tensor), y.dtype, y.device)
        elif isinstance(x, int) and isinstance(y, int):
            return RingTensor(torch.where(condition.tensor.bool(), x, y).to(data_type), condition.dtype,
                              condition.device)

    @staticmethod
    def random(shape, dtype='int', device=DEVICE, down_bound=-HALF_RING, upper_bound=HALF_RING - 1):
        v = torch.randint(down_bound, upper_bound, shape, dtype=data_type, device=device)
        return RingTensor(v, dtype, device)

    @staticmethod
    def empty(size, dtype='int', device=DEVICE):
        return RingTensor(torch.empty(size, dtype=data_type), dtype, device)

    @staticmethod
    def empty_like(tensor):
        if isinstance(tensor, RingTensor):
            return RingTensor(torch.empty_like(tensor.tensor), tensor.dtype, tensor.device)
        else:
            raise TypeError("unsupported operand type(s) for empty_like 'RingTensor' and ", type(tensor))

    @staticmethod
    def zeros(size, dtype='int', device=DEVICE):
        return RingTensor(torch.zeros(size, dtype=data_type, device=device), dtype)

    @staticmethod
    def zeros_like(tensor, dtype='int', device=DEVICE):
        if isinstance(tensor, RingTensor):
            return RingTensor(torch.zeros_like(tensor.tensor), tensor.dtype, tensor.device)
        elif isinstance(tensor, torch.Tensor):
            return RingTensor(torch.zeros_like(tensor), dtype, device)
        else:
            raise TypeError("unsupported operand type(s) for zeros_like 'RingTensor' and ", type(tensor))

    @staticmethod
    def ones(size, dtype='int', device=DEVICE):
        scale = DTYPE_SCALE_MAPPING[dtype]
        return RingTensor(torch.ones(size, dtype=data_type, device=device) * scale, dtype)

    @staticmethod
    def ones_like(tensor, dtype='int', device=DEVICE):
        if isinstance(tensor, RingTensor):
            return RingTensor(torch.ones_like(tensor.tensor) * tensor.scale, tensor.dtype, tensor.device)
        elif isinstance(tensor, torch.Tensor):
            scale = DTYPE_SCALE_MAPPING[dtype]
            return RingTensor(torch.ones_like(tensor) * scale, dtype, device)
        else:
            raise TypeError("unsupported operand type(s) for ones_like 'RingTensor' and ", type(tensor))

    @staticmethod
    def full(size, fill_value, device=DEVICE):
        return RingTensor.convert_to_ring(torch.full(size, fill_value, device=device))

    @staticmethod
    def full_like(tensor, fill_value, device=DEVICE):
        if isinstance(tensor, RingTensor):
            return RingTensor.full(tensor.shape, fill_value, tensor.device)
        elif isinstance(tensor, torch.Tensor):
            return RingTensor.convert_to_ring(torch.full_like(tensor, fill_value, device=device))
        else:
            raise TypeError("unsupported operand type(s) for full_like 'RingTensor' and ", type(tensor))

    @staticmethod
    def arange(start, end, step=1, dtype='int', device=DEVICE):
        return RingTensor(torch.arange(start, end, step, dtype=data_type, device=device), dtype)
