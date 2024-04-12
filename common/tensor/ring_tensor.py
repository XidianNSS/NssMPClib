from functools import singledispatchmethod

import torch.nn.functional as F

from common.utils import *
from config.base_configs import BIT_LEN, DEVICE, data_type, DTYPE_MAPPING, DTYPE_SCALE_MAPPING


class RingTensor(object):
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
    def from_tensor(self, tensor, dtype='int', device=DEVICE):
        """Encapsulate a torch.Tensor as a RingTensor.
        """
        self.tensor = tensor.to(device)
        self.dtype = dtype
        self.bit_len = BIT_LEN

    @__init__.register(int)
    @__init__.register(list)
    def from_item(self, item, dtype='int', device=DEVICE):
        """
        Encapsulate an int or list item as a RingTensor.
        """
        self.tensor = torch.tensor(item, dtype=data_type, device=device)
        self.dtype = dtype
        self.bit_len = BIT_LEN

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
        return self.__class__(new_value, self.dtype, self.device)

    def __str__(self):
        return f"{self.__class__.__name__}\n value:{self.tensor} \n dtype:{self.dtype} \n scale:{self.scale}"

    def __getitem__(self, item):
        return self.__class__(self.tensor[item], self.dtype).clone()

    def __setitem__(self, key, value):
        if isinstance(value, RingTensor):
            self.tensor[key] = value.tensor.clone()
        else:
            raise TypeError(f"unsupported operand type(s) for setitem '{type(self)}' and {type(value)}")

    def __len__(self):
        return len(self.tensor)

    def __invert__(self):
        return self.__class__(~self.tensor, self.dtype, self.device)

    def __add__(self, other):
        if isinstance(other, RingTensor):
            new_value = self.tensor + other.tensor
        elif isinstance(other, torch.Tensor) or isinstance(other, int):
            new_value = self.tensor + other
        else:
            raise TypeError("unsupported operand type(s) for + 'RingTensor' and ", type(other),
                            'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, RingTensor):
            new_value = self.tensor - other.tensor
        elif isinstance(other, (int, torch.Tensor)):
            new_value = self.tensor - other
        else:
            raise TypeError(
                "unsupported operand type(s) for - 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, RingTensor):
            assert self.dtype == other.dtype, "dtype not equal"
            new_value = (self.tensor * other.tensor) // self.scale
        elif isinstance(other, int):
            new_value = self.tensor * other
        else:
            raise TypeError(
                "unsupported operand type(s) for * 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mod__(self, other):
        if isinstance(other, int):
            new_value = (self.tensor % other)
        else:
            raise TypeError(
                "unsupported operand type(s) for % 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

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
            item = torch.tensor(item, dtype=data_type, device=DEVICE)
        assert isinstance(item, torch.Tensor), f"unsupported data type(s): {type(item)}"
        scale = DTYPE_MAPPING[item.dtype]
        v = torch.round(item * scale) if scale != 1 else item
        dtype = 'int' if scale == 1 else 'float'
        r = cls(v.to(data_type), dtype=dtype, device=item.device)
        return r

    @classmethod
    def load_from_file(cls, file_path):
        """
        Loads a RingTensor from a file
        """
        return cls(torch.load(file_path))

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
        assert all([t.dtype == tensor_list[0].dtype for t in tensor_list]), "The element type should be the same"
        if isinstance(tensor_list[0], RingTensor):
            def fast_concat(tensors, dim=0):
                shape = list(tensors[0].shape)
                shape[dim] = sum([t.shape[dim] for t in tensors])
                result = torch.empty(*shape, dtype=tensors[0].dtype, device=tensors[0].device)
                offset = 0
                for t in tensors:
                    size = t.shape[dim]
                    result.narrow(dim, offset, size).copy_(t)
                    offset += size
                return result

            return cls(fast_concat([t.tensor for t in tensor_list], dim), tensor_list[0].dtype)
        else:
            raise TypeError(f"unsupported operand type(s) for cat '{cls.__name__}' and {type(tensor_list[0])}")

    @classmethod
    def stack(cls, tensor_list, dim=0):
        assert all([t.dtype == tensor_list[0].dtype for t in tensor_list]), "The element type should be the same"
        return cls(torch.stack([t.tensor for t in tensor_list], dim), tensor_list[0].dtype, tensor_list[0].device)

    @classmethod
    def roll(cls, input, shifts, dims=0):
        return cls(torch.roll(input.tensor, shifts=shifts, dims=dims), input.dtype, input.device)

    @classmethod
    def rotate(cls, input, shifts):
        """
        For the input 2-dimensional RingTensor, rotate each row to the left or right
        """
        if isinstance(shifts, list):
            shifts = torch.tensor(shifts, dtype=data_type, device=input.device)

        n = input.shape[1]
        rows = torch.arange(input.shape[0]).view(-1, 1)
        indices = (torch.arange(n, device=shifts.device) + shifts.view(-1, 1)) % n
        result = input[rows, indices]

        return result

    def convert_to_real_field(self):
        """
        convert the tensor from ring field to real field
        """
        return self.tensor / self.scale

    def sum(self, dim=0):
        new_value = torch.sum(self.tensor, dim=dim, dtype=self.tensor.dtype)
        return self.__class__(new_value, dtype=self.dtype, device=self.device)

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
        torch.save(self.tensor, file_path)

    def clone(self):
        clone = self.__class__(self.tensor.clone(), dtype=self.dtype, device=self.device)
        clone.bit_len = self.bit_len
        return clone

    def get_bit(self, item):
        """
        get a bit from the RingTensor
        """
        assert (self.bit_len >> item >= 0), "bit index out of range"
        # return (self.tensor >> item) & 1
        return RingTensor((self.tensor >> item) & 1)

    def reshape(self, *shape):
        new = self.__class__(self.tensor.reshape(*shape), dtype=self.dtype, device=self.device)
        new.bit_len = self.bit_len
        return new

    def img2col(self, k_size: int, stride: int):
        """
        Img2Col algorithm
        Speed up convolution and pooling

        Args:
            k_size: the size of kernel
            stride: the stride of the convolution and pooling

        Returns:
            col: the RingTensor object
            batch
            out_size: the size of the col
            channel
        """

        img = self.tensor

        batch, channel, height, width = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
        out_h = (height - k_size) // stride + 1
        out_w = (width - k_size) // stride + 1
        kw = kh = k_size
        out_size = out_w * out_h
        col = torch.zeros(size=(batch, channel, kw * kh, out_size), dtype=data_type, device=self.device)
        for y in range(out_h):
            y_start = y * stride
            y_end = y_start + kh
            for x in range(out_w):
                x_start = x * stride
                x_end = x_start + kw
                col[..., 0:, y * out_w + x] = img[..., y_start:y_end, x_start:x_end].reshape(batch, channel, kh * kw)

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

    def squeeze(self, dim):
        return self.__class__(self.tensor.squeeze(dim), dtype=self.dtype, device=self.device)

    def unsqueeze(self, dim):
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

    def permute(self, dims):
        new = self.__class__(self.tensor.permute(dims=dims), dtype=self.dtype, device=self.device)
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
