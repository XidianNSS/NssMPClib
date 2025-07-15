#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

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
    Define the tensor on the ring. Support for the part of operations on pytorch tensor

    ATTRIBUTES:
        * **tensor** (*torch.Tensor*): the pytorch tensor
        * **dtype** (*Type*) : the dtype of the RingTensor
        * **bit_len** (*int*): the expected length of the binary representation after conversion

    """

    @singledispatchmethod
    def __init__(self):
        """
        Initialize method.

        Using **@singledispatchmethod** allows you to select different initialization logic depending on the input type.
        """
        self.bit_len = None
        self.dtype = None
        self.tensor = None

    @__init__.register(torch.Tensor)
    def from_tensor(self, tensor, dtype='int', device=DEVICE, bit_len=BIT_LEN):
        self.tensor = tensor.to(device) if tensor.device.type != device else tensor
        self.dtype = dtype
        self.bit_len = bit_len

    @__init__.register(int)
    @__init__.register(list)
    def from_item(self, item, dtype='int', device=DEVICE, bit_len=BIT_LEN):
        self.tensor = torch.tensor(item, dtype=data_type, device=device)
        self.dtype = dtype
        self.bit_len = bit_len

    @property
    def scale(self):
        """
        Return the scale of the RingTensor.

        Obtain the corresponding value from *DTYPE_SCALE_MAPPING* based on dtype.
        """
        return DTYPE_SCALE_MAPPING[self.dtype]

    @property
    def device(self):
        """
        :return: the device type of the current RingTensor, expressed as a string
        :rtype: str
        """
        return self.tensor.device.__str__()

    @property
    def shape(self):
        """
        :return: RingTensor shape information.
        :rtype: torch.Size
        """
        return self.tensor.shape

    @property
    def T(self):
        """
        Create a new RingTensor instance with the old dtype, device, and bit_len

        :return: Transpose of the current RingTensor
        :rtype: RingTensor
        """
        new_value = self.tensor.T
        return self.__class__(new_value, self.dtype, self.device, self.bit_len)

    def __str__(self):
        """
        A custom string representation that provides class name, RingTensor value, data type, and scaling information.

        :returns: A string that represents the RingTensor instance.
        :rtype: str
        """
        return f"{self.__class__.__name__}\n value:{self.tensor} \n dtype:{self.dtype} \n scale:{self.scale}"

    def __getitem__(self, item):
        """
        Support index access

        :param item: the index
        :type item: Int or slice or list or tuple, and so on.
        :return: A new RingTensor instance containing the indexed part.
        :rtype: RingTensor
        """
        return self.__class__(self.tensor[item], self.dtype, self.device, self.bit_len)

    def __setitem__(self, key, value):
        """
        Support index assignment

        :param key: the index
        :type key: Int or slice or list or tuple, and so on.
        :param value: The value corresponding to the index
        :type value: RingTensor
        :raises TypeError: The value passed in is not of the RingTensor type.
        """
        if isinstance(value, RingTensor):
            self.tensor[key] = value.tensor.clone()
        else:
            raise TypeError(f"unsupported operand type(s) for setitem '{type(self)}' and {type(value)}")

    def __len__(self):
        """
        :return: The length of the current RingTensor.
        :rtype: int
        """
        return len(self.tensor)

    def __invert__(self):
        """
        Implement the reverse operation by bit.

        :return: A new RingTensor representing the bitwise inversion.
        :rtype: RingTensor
        """
        return self.__class__(~self.tensor, self.dtype, self.device, self.bit_len)

    def __add__(self, other):
        """
        Adds a RingTensor object with a corresponding type.

        The type of other is determined first:
            * If *other* is *RingTensor*: perform RingTensor addition and update bit_len to the maximum of the two.
            * If *other* is *torch.Tensor* or *int*: add directly

        :param other: The object to be added to self.
        :type other: RingTensor or torch.Tensor or int
        :return: The result of the addition
        :rtype: RingTensor
        :raises TypeError: If *other* is not a supported type.
        """
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
        """
        Perform right-hand addition for an RingTensor instance.
        """
        return self.__add__(other)

    def __iadd__(self, other):
        """
        In situ addition operation, using the **+=** operator.

        Similar to :meth:`__add__`, but directly modifies the value of *self.tensor*. Also check the type and process, update *self.tensor*.

        """
        if isinstance(other, RingTensor):
            self.tensor += other.tensor
            self.bit_len = max(self.bit_len, other.bit_len)
        elif isinstance(other, torch.Tensor) or isinstance(other, int):
            self.tensor += other
        else:
            raise TypeError("unsupported operand type(s) for += 'RingTensor' and ", type(other))
        return self

    def __sub__(self, other):
        """
        Subtracts a RingTensor object with a corresponding type.

        The type of other is determined first:
            * If *other* is *RingTensor* : perform RingTensor subtraction and update bit_len to the maximum of the two.
            * If *other* is *torch.Tensor* or *int* : subtract directly

        :param other: subtrahend.
        :type other: RingTensor or torch.Tensor or int
        :return: The result of the subtraction
        :rtype: RingTensor
        :raises NotImplemented: If *other* is not a supported type.
        """
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
        """
        Perform right-hand subtraction for an RingTensor instance.
        """
        return -self.__sub__(other)

    def __isub__(self, other):
        """
        Perform a subtraction operation in place, using the **-=** operator.
        """
        if isinstance(other, RingTensor):
            self.tensor -= other.tensor
            self.bit_len = max(self.bit_len, other.bit_len)
        elif isinstance(other, torch.Tensor) or isinstance(other, int):
            self.tensor -= other
        else:
            raise TypeError("unsupported operand type(s) for -= 'RingTensor' and ", type(other))
        return self

    def __mul__(self, other):
        """
        This method implements the standard multiplication operation.

        The type of other is determined first:
            * If *other* is *RingTensor* : Calculate the new value based on the scale property of both objects and select the appropriate dtype
            * If *other* is *int* : multiply directly
            * If *other* is not a supported type : return **NotImplemented**

        :param other: multiplier
        :type other: RingTensor or int
        :return: Multiplication result
        :rtype: RingTensor
        :raises NotImplemented: If *other* is not a supported type.
        """
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
        """
        Right-hand multiplication operation
        """
        return self.__mul__(other)

    def __imul__(self, other):
        """
        Local multiplication operation, using the ***=** operator.
        """
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
        """
        Implement modular operation **%**.

        * If *other* is an integer, 'self.tensor % other' is evaluated, then calculate the new *bit_len*.
            * If *self.bit_len* is greater than the logarithm of *other*, its pair value is used.
            * otherwise *self.bit_len* is used.

        :param other: modulo
        :type other: int
        :return: The result after taking the mold
        :rtype: RingTensor
        :raises TypeError: If *other* is not an integer.
        """
        if isinstance(other, int):
            new_value = (self.tensor % other)
            bit_len = math.ceil(math.log2(other)) if self.bit_len > math.log2(other) else self.bit_len
        else:
            raise TypeError(
                "unsupported operand type(s) for % 'RingTensor' and ", type(other), 'only support int type')
        return self.__class__(new_value, self.dtype, self.device, bit_len)

    def __matmul__(self, other):
        """
        Implement matrix multiplication **@**.

        The type of *other* is determined first, if the *other* is not RingTensor, raise **TypeError**. When *other*
        is a RingTensor, First, make sure *self* and *other* have the same data type. If not, an assertion error is
        thrown, then check whether the device is a CUDA device:
            * If on a CUDA device, call the cuda_matmul function for matrix multiplication.
            * If the device is not CUDA, the matrix multiplication of the CPU is performed.

        :param other: multiplier
        :type other: RingTensor
        :return: the result of matrix multiplication
        :rtype: RingTensor
        :raises TypeError: If *other* is not a RingTensor.
        """
        if isinstance(other, RingTensor):
            assert self.dtype == other.dtype, "dtype not equal"
            if 'cuda' in self.device:
                new_value = cuda_matmul(self.tensor, other.tensor) // self.scale
            else:
                new_value = torch.matmul(self.tensor, other.tensor) // self.scale
        else:
            return NotImplemented
        return self.__class__(new_value, self.dtype, self.device)

    def __truediv__(self, other):
        """
        Implement true division **/**.

        The type of other is determined first:
            * If *other* is *RingTensor* : Then check whether the scale is the same and perform the calculation.
            * If *other* is *torch.Tensor* or *int* : divide directly

        .. note::
            Use torch.round() to process the final value, ensuring that the result returned is an integer.

        :param other: divisor
        :type other: RingTensor or tensor or int
        :return: quotient
        :rtype: RingTensor
        :raises TypeError: If *other* is not a supported type.
        """
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
        """
        Implement proper division in place **/=**.

        Similar to :meth:`__truediv__`, but directly modifies the value of *self.tensor*. Also check the type and process, update *self.tensor*.
        """
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
        """
        Implement floor division **//**.

        The processing is similar to :py:meth:`__truediv__`, but uses **//** to perform floor division.

        :param other: divisor
        :type other: RingTensor or torch.Tensor or int
        :return: quotient
        :rtype: RingTensor
        :raises TypeError: If *other* is not a supported type.
        """
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
        """
        Implement local floor division operations **//=**.

        Similar to :py:meth:`__floordiv__`,  but modifying *self.tensor* directly, perform floor division and update the value
        """
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
        """
        Implement the unary minus operation **-**.

        Take the negative of each element in *self.tensor*, at the shame time, keep the same data types and devices.

        :return: The negative number of each element.
        :rtype: RingTensor
        """
        new_value = -self.tensor
        return self.__class__(new_value, dtype=self.dtype, device=self.device)

    def __eq__(self, other):
        """
        Implement equal operation **==**.

        The type of other is determined first:
            * If *other* is *RingTensor* : Compare two tensors to see if they are equal, and the result is stored in new_value.
            * If *other* is *int* : Multiply this by *self.scale* for comparison.

        :param other: The object to compare with *self*
        :type other: RingTensor or int
        :return: Boolean comparison results
        :rtype: RingTensor
        :raises TypeError: If *other* is not a supported type.
        """
        if isinstance(other, RingTensor):
            new_value = (self.tensor == other.tensor)
        elif isinstance(other, int):
            new_value = (self.tensor == other * self.scale)
        else:
            raise TypeError(
                "unsupported operand type(s) for == 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    def __ne__(self, other):
        """
        Implementation is not equal to operation **!=**.

        Similar to the :py:meth:`__eq__` method, compare the inequality of two tensors.

        :param other: The object to compare with *self*
        :type other: RingTensor or int
        :return: Boolean comparison results
        :rtype: RingTensor
        :raises TypeError: If *other* is not a supported type.
        """
        if isinstance(other, RingTensor):
            new_value = self.tensor != other.tensor
        elif isinstance(other, int):
            new_value = self.tensor != other * self.scale
        else:
            raise TypeError(
                "unsupported operand type(s) for != 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    def __gt__(self, other):
        """
        Implement greater-than operation **>**.

        The type of other is determined first:
            * If *other* is *RingTensor* : Compares the elements of the two tensors and generates the Boolean array new_value.
            * If *other* is *int* : Multiply this by *self.scale* for comparison.

        :param other: The object to compare with *self*
        :type other: RingTensor or int
        :return: Boolean comparison results
        :rtype: RingTensor
        :raises TypeError: If *other* is not a supported type.
        """
        if isinstance(other, RingTensor):
            new_value = (self.tensor > other.tensor)
        elif isinstance(other, int):
            new_value = (self.tensor > other * self.scale)
        else:
            raise TypeError(
                "unsupported operand type(s) for > 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    def __ge__(self, other):
        """
        Implement greater-than-equal operations **>=**.

        Similar to :py:meth:`__gt__`, but with greater than or equal comparison symbols.

        :param other: The object to compare with *self*
        :type other: RingTensor or int
        :return: Boolean comparison results
        :rtype: RingTensor
        :raises TypeError: If *other* is not a supported type.
        """
        if isinstance(other, RingTensor):
            new_value = (self.tensor >= other.tensor)
        elif isinstance(other, int):
            new_value = (self.tensor >= other * self.scale)
        else:
            raise TypeError(
                "unsupported operand type(s) for >= 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    def __lt__(self, other):
        """
        Implement the less than operation **<**.

        Similar to :py:meth:`__gt__`, the resulting Boolean array new_value reflects less than the result of the comparison.

        :param other: The object to compare with *self*
        :type other: RingTensor or int
        :return: Boolean comparison results
        :rtype: RingTensor
        :raises TypeError: If *other* is not a supported type.
        """
        if isinstance(other, RingTensor):
            new_value = (self.tensor < other.tensor)
        elif isinstance(other, int):
            new_value = (self.tensor < other * self.scale)
        else:
            raise TypeError(
                "unsupported operand type(s) for < 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    def __le__(self, other):
        """
        Implements the less than or equal operation **<=**.

        Similar to :py:meth:`__ge__`, the type is judged and a less than or equal comparison is made/

        :param other: The object to compare with *self*
        :type other: RingTensor or int
        :return: Boolean comparison results
        :rtype: RingTensor
        :raises TypeError: If *other* is not a supported type.
        """
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
        """
        Implement bitwise XOR operations **^**.

        The type of other is determined first:
            * If *other* is *RingTensor* : Bitwise xOR operations using tensors.
            * If *other* is *torch.Tensor* or *int* : Also do XOR by bit.

        :param other: Object that makes a bit-by-bit xOR with *self*
        :type other: RingTensor or int or torch.Tensor
        :return: Boolean comparison results
        :rtype: RingTensor
        :raises TypeError: If *other* is not a supported type.
        """
        if isinstance(other, RingTensor):
            return self.__class__(self.tensor ^ other.tensor, self.dtype, self.device)
        elif isinstance(other, (int, torch.Tensor)):
            return self.__class__(self.tensor ^ other, self.dtype, self.device)
        else:
            raise TypeError(
                "unsupported operand type(s) for ^ 'RingTensor' and ", type(other), 'please convert to ring first')

    def __or__(self, other):
        """
        Implement bitwise or operations **|**.

        The type of other is determined first:
            * If *other* is *RingTensor* : Perform bitwise or operations.
            * If *other* is *int* : Perform the corresponding bitwise or operation.

        :param other: Object that makes a bitwise or with *self*
        :type other: RingTensor or int
        :return: Boolean comparison results
        :rtype: RingTensor
        :raises TypeError: If *other* is not a supported type.
        """
        if isinstance(other, RingTensor):
            return self.__class__(self.tensor | other.tensor, self.dtype, self.device)
        elif isinstance(other, int):
            return self.__class__(self.tensor | other, self.dtype, self.device)
        else:
            raise TypeError(
                "unsupported operand type(s) for | 'RingTensor' and ", type(other), 'please convert to ring first')

    def __and__(self, other):
        """
        Implement bit-and-operate **&**

        The type of other is determined first:
            * If *other* is *RingTensor* : Perform bitwise and operations.
            * If *other* is *int* : Execute by bit and.

        :param other: Object that makes a bitwise and with *self*
        :type other: RingTensor or int
        :return: Boolean comparison results
        :rtype: RingTensor
        :raises TypeError: If *other* is not a supported type.
        """
        if isinstance(other, RingTensor):
            return self.__class__(self.tensor & other.tensor, self.dtype, self.device)
        elif isinstance(other, int):
            return self.__class__(self.tensor & other, self.dtype, self.device)
        else:
            raise TypeError(
                "unsupported operand type(s) for & 'RingTensor' and ", type(other), 'please convert to ring first')

    def __rshift__(self, other):
        """
        Right shift operation **>>**.

        The type of other is determined first:
            * If *other* is *RingTensor* : Move the value of the *self* object to the right by *other* bits.
            * If *other* is *int* : Also perform the right shift

        :param other: Number of bits shifted to the right
        :type other: RingTensor or int
        :return: The result of the *self* shift
        :rtype: RingTensor
        :raises TypeError: If *other* is not a supported type.
        """
        if isinstance(other, RingTensor):
            return self.__class__(self.tensor >> other.tensor, self.dtype, self.device)
        elif isinstance(other, int):
            return self.__class__(self.tensor >> other, self.dtype, self.device)

        else:
            raise TypeError(
                "unsupported operand type(s) for >> 'RingTensor' and ", type(other), 'please convert to ring first')

    def __lshift__(self, other):
        """
        Implement left shift operation **<<**.

        Similar to :meth:`__rshift__`, but the operation is to move left.

        :param other: Number of bits shifted to the left
        :type other: RingTensor or int
        :return: The result of the *self* shift
        :rtype: RingTensor
        :raises TypeError: If *other* is not a supported type.
        """
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
        Converts the item to RingTensor.

        If the *item* is an integer, a floating point number, or a list, use *torch.tensor* to convert it to a
        PyTorch tensor.Then assert whether the converted item is of type torch.Tensor. Get the data type scaling
        factor scale for item.
            * If scale is not equal to 1, round the item and multiply it by scale, otherwise leave it unchanged.
            * Set dtype to *int* or *float* depending on the value of scale.

        :param item: item to convert
        :type item: int or float or list or torch.Tensor
        :returns: A RingTensor object converted from item.
        :rtype: RingTensor
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
        # TODO: 返回的是tensor，而非RingTensor;改为RingTensor而类型无法确定。
        """
        Loads a RingTensor from a file.

        :param file_path: The path of the loaded file
        :type file_path: str
        :return: The contents stored in the file.
        :rtype: torch.Tensor or torch.nn.Module
        """
        return torch.load(file_path)

    @classmethod
    def mul(cls, x, y):
        """
        Multiplies two RingTensors without truncation.

        This only applies if *x* and *y* are both RingTensor.

        :param x: multiplicand
        :type x: RingTensor
        :param y: multiplier
        :type y: RingTensor
        :returns: Multiplication result
        :rtype: RingTensor
        """
        return cls(x.tensor * y.tensor, x.dtype)

    @classmethod
    def matmul(cls, x, y):
        """
        Matrix Multiplies two RingTensors without truncation.

        First, check the device type of x and y:
            * If the device is a CPU, use torch.matmul to compute matrix multiplication.
            * If the device is a CUDA, call the cuda_matmul function for matrix multiplication.

        :param x: multiplicand
        :type x: RingTensor
        :param y: multiplier
        :type y: RingTensor
        :returns: The result of matrix multiplication
        :rtype: RingTensor
        :raises TypeError: x and y are not on the same device.
        """
        if x.device != y.device:
            raise TypeError(
                "Expected all ring tensors to be on the same device, but found at least two devices,"
                + f" {x.device} and {y.device}!")
        if x.device == 'cpu':
            return cls(torch.matmul(x.tensor, y.tensor), x.dtype)
        if 'cuda' in x.device:
            return cls(cuda_matmul(x.tensor, y.tensor), x.dtype)

    @classmethod
    def cat(cls, tensor_list, dim=0):
        """
        Concatenate multiple RingTensor objects in the specified dimension.

        .. note::
            All RingTensor to be spliced must have the same shape in dimensions other than the spliced dimension. If this condition is not met, the method throws an error.

        :param tensor_list: A list of RingTensor to splice.
        :type tensor_list: list or tuple or RingTensor
        :param dim: Concatenated dimensions (default is 0)
        :type dim: int
        :return: The result after splicing
        :rtype: RingTensor
        """
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
        """
        Stack multiple RingTensor objects on a specified dimension.

        First, Use assert to ensure that all RingTensor data types in *tensor_list* are the same. If not,
        throw an error. Then use torch.stack to stack tensor_list tensors in the specified dimension.

        :param tensor_list: A list of RingTensor to splice.
        :type tensor_list: list or tuple
        :param dim: stack dimensions (default is 0)
        :type dim: int
        :return: The result after stacking
        :rtype: RingTensor
        """
        assert all([t.dtype == tensor_list[0].dtype for t in tensor_list]), "The element type should be the same"
        return cls(torch.stack([t.tensor for t in tensor_list], dim), tensor_list[0].dtype, tensor_list[0].device,
                   tensor_list[0].bit_len)

    @classmethod
    def gather(cls, input, dim, index):
        """
        Collect data from the input RingTensor according to the index.

        Use *torch.gather* to collect data from *input.tensor* on the specified dimension to form a new RingTensor.

        :param input: RingTensor to operate
        :type input: RingTensor
        :param dim: Dimensions to collect
        :type dim: int
        :param index: Index tensor
        :type index: RingTensor or torch.Tensor
        :return: Results collected according to the index.
        :rtype: RingTensor
        """
        if isinstance(index, RingTensor):
            index = index.tensor
        return cls(torch.gather(input.tensor, dim, index), input.dtype, input.device, input.bit_len)

    @classmethod
    def roll(cls, input, shifts, dims=0):
        """
        Scroll the input RingTensor.

        Roll shift using *torch.roll*.

        :param input: RingTensor object to be operated.
        :type input: RingTensor
        :param shifts: The amount of displacement of rolling
        :type shifts: int
        :param dims: Dimensions to roll
        :type dims: int
        :return: *input* after rolling shift
        :rtype: RingTensor
        """
        return cls(torch.roll(input.tensor, shifts=shifts, dims=dims), input.dtype, input.device, input.bit_len)

    @classmethod
    def rotate(cls, input, shifts):
        """
        For a two-dimensional RingTensor of input, shift each row to the left or right depending on the corresponding element in shifts.

        Ensure that the shifts data type is a tensor, If shifts is a list or RingTensor,Convert it to *torch.tensor*.
            * if shifts > 0, rotate to the right
            * if shifts < 0, rotate to the left

        :param input: RingTensor object to be operated.
        :type input: RingTensor
        :param shifts: Rotational displacement
        :type shifts: RingTensor or list or torch.Tensor.
        :return: Input after rotation operation
        :rtype: RingTensor
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
        Creates a one-hot RingTensor from input RingTensor.

        Use *torch.nn.functional.one_hot* for a heat encoding and multiply by the input scaling factor.

        :param input: the indices
        :type input: RingTensor
        :param num_classes: the number of classes
        :type num_classes: int

        :return: A one-hot RingTensor
        :rtype: RingTensor
        """
        return cls(F.one_hot(input.tensor.to(torch.int64), num_classes).to(data_type) * input.scale, input.dtype,
                   input.device)

    @classmethod
    def exp(cls, x):
        """
        Calculate the exponent of the input RingTensor.

        Calculate using *torch.exp* after converting x to the real number field.The exponent based on e is (e^x).

        :param x: RingTensor to be calculated
        :type x: RingTensor
        :return: Input index
        :rtype: RingTensor
        """
        assert x.dtype == 'float', "The input tensor should be float type"
        return cls.convert_to_ring(torch.exp(x.convert_to_real_field()))

    @classmethod
    def exp2(cls, x):
        """
        Compute the base 2 exponent of the input RingTensor. The exponent with base 2 is (2^x).

        This method is similar to :meth:`exp`, but *exp2* is an exponent with base 2

        :param x: RingTensor to be calculated
        :type x: RingTensor
        :return: Input index
        :rtype: RingTensor
        """
        return cls.convert_to_ring(torch.exp2(x.convert_to_real_field()))

    @classmethod
    def tanh(cls, x):
        """
        Calculate the double of the input RingTensor x.

        First, use the :meth:`convert_to_real_field()` function to convert x from a ring domain to the real number domain.

        .. note::
            The function returns values within the range of [-1, 1].

        :param x: RingTensor to be calculated
        :type x: RingTensor
        :return: Calculation result
        :rtype: RingTensor
        """
        return cls.convert_to_ring(torch.tanh(x.convert_to_real_field()))

    def convert_to_real_field(self):
        """
        convert the RingTensor from ring field to real field.

        RingTensor types mapped to PyTorch depending on the bit length (**BIT_LEN**) and data type (**dtype**) to get *torch_type*.

        :return: The torch.Tensor with numerical values of the real number field.
        :rtype: Tensor
        """
        torch_type = {64: {'int': torch.int64, 'float': torch.float64},
                      32: {'int': torch.int32, 'float': torch.float32}}
        return (self.tensor / self.scale).to(torch_type[BIT_LEN][self.dtype])

    def convert_to_range(self, bit_len):
        """
        Convert the RingTensor to the ring of a specified bit length.

        :param bit_len: The specified bit length
        :type bit_len: int
        :return: The RingTensor after conversion
        :rtype: RingTensor
        """
        mod = 2 ** bit_len
        half_ring = 2 ** (bit_len - 1)

        self.tensor %= mod
        self.tensor = torch.where(self.tensor >= half_ring, self.tensor - mod, self.tensor)
        self.bit_len = bit_len
        return self

    def sum(self, dim=0):
        """
        Sums tensors over a specified dimension.

        Use *torch.sum* to calculate the sum of *self.tensor* along the specified dimension dim, while retaining the RingTensor's original data type.

        :param dim: Specified Dimension
        :type dim: int
        :return: Sum result
        :rtype: RingTensor
        """
        new_value = torch.sum(self.tensor, dim=dim, dtype=self.tensor.dtype)
        return self.__class__(new_value, dtype=self.dtype, device=self.device, bit_len=self.bit_len)

    def all(self):
        """
        Check whether all elements of the RingTensor are true.

        :return: Check the bool value of the result
        :rtype: bool
        """
        return self.tensor.all()

    def any(self):
        """
        Check if the RingTensor has at least one element that is true.

        :return: Check the bool value of the result
        :rtype: bool
        """
        return self.tensor.any()

    def to(self, target):
        """
        Move the RingTensor to a specified device or convert the RingTensor to a specified type.

        :param target: The specified computing device.
        :type target: str or dtype
        :return: The RingTensor after the move or convert
        :rtype: RingTensor

        Args:
            target:
        """
        if target == 'int' and self.dtype == 'float':
            self.tensor = self.tensor // DTYPE_SCALE_MAPPING['float']
            self.dtype = 'int'
        elif target == 'float' and self.dtype == 'int':
            self.tensor = self.tensor * DTYPE_SCALE_MAPPING['float']
            self.dtype = 'float'
        else:
            self.tensor = self.tensor.to(target)
        return self

    def cpu(self):
        """
        Move the RingTensor to the CPU device.

        :return: The RingTensor after the move
        :rtype: RingTensor
        """
        self.tensor = self.tensor.cpu()
        return self

    def cuda(self, device=None):
        """
        Move the RingTensor to the GPU device.If device is specified, the device is used.

        :param device: The specified computing device.
        :type device: str
        :return: The RingTensor after the move
        :rtype: RingTensor
        """
        self.tensor = self.tensor.cuda(device)
        return self

    def save(self, file_path):
        """
        Saves the current RingTensor instance to the specified file path.

        Serialize the self object and save it to a file using *torch.save*

        :param file_path: The path of the saved file
        :type file_path: str
        """
        torch.save(self, file_path)

    def clone(self):
        """
        Creates a clone of the current RingTensor.

        Independent manipulation or modification in cases where it is necessary to keep the original RingTensor unchanged.

        :return: New clone
        :rtype: RingTensor
        """
        clone = self.__class__(self.tensor.clone(), dtype=self.dtype, device=self.device)
        clone.bit_len = self.bit_len
        return clone

    def get_bit(self, item):
        """
        Retrieves a bit from the specified location.

        :param item: The index of the bit to be extracted
        :type item: int
        :returns: The extracted bit
        :rtype: RingTensor
        """
        assert (self.bit_len >> item >= 0), "bit index out of range"
        return RingTensor((self.tensor >> item) & 1)

    def get_tensor_bit(self, item):
        """
        get a bit from the RingTensor

        This method is similar to :meth:`get_bit`, except that the extracted result type is torch

        :param item: The index of the bit to be extracted
        :type item: int
        :returns: The extracted bit
        :rtype: torch.Tensor
        """
        assert (self.bit_len >> item >= 0), "bit index out of range"
        return (self.tensor >> item) & 1

    def reshape(self, *shape):
        """
        Reshape the current RingTensor instance.

        The method is used to reshape the current RingTensor, Create a new RingTensor instance new using the reshaped
        RingTensor and the current instance's dtype and device attributes.

        :param shape: A variable number of parameters, specifying a new shape.
        :type shape: torch.Size or tuple
        :return: *self* after changing shape
        :rtype: RingTensor
        """
        new = self.__class__(self.tensor.reshape(*shape), dtype=self.dtype, device=self.device)
        new.bit_len = self.bit_len
        return new

    def img2col(self, k_size: int, stride: int):
        """
        Img2Col algorithm using PyTorch's unfold to speed up convolution and pooling

        First, determine the shape of the img RingTensor, and retrieve the batch size, number of channels, height,
        and width. Then,The height and width of the output feature map are calculated based on the size of the input
        image, the size of the convolutional kernel, and the stride.

        Use PyTorch's unfold method to unfold the image on the specified dimensions (height and width) to get a
        RingTensor that contains all the convolution Windows, Then rearrange the dimensions and change the RingTensor shape.

        :param k_size: the size of the kernel
        :type k_size: int
        :param stride: the stride of the convolution and pooling
        :type stride: int
        :returns: the RingTensor object, the batch size of the image, the size of the col, The number of channels to input the image.
        :rtype: tuple
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
        """
        Repeat elements of the input RingTensor along a specified dimension.

        The method uses *torch.repeat_interleave* to repeat each element in the RingTensor according to the
        specified number of repeats along the given dimension.

        :param repeats: The number of repetitions for each element.
        :type repeats: int or list of int
        :param dim: The dimension along which to repeat the elements.
        :type dim: int
        :return: A new RingTensor with the repeated elements.
        :rtype: RingTensor
        """
        return self.__class__(self.tensor.repeat_interleave(repeats, dim), dtype=self.dtype, device=self.device)

    def repeat(self, *sizes):
        """
        Repeat the input RingTensor along each dimension specified by sizes.

        This method uses *torch.repeat* to replicate the RingTensor according to the provided sizes for each dimension.

        :param sizes: The number of times to repeat the RingTensor along each dimension.
        :type sizes: torch.Size or int
        :return: A new RingTensor with the repeated elements.
        :rtype: RingTensor
        """
        return self.__class__(self.tensor.repeat(*sizes), dtype=self.dtype, device=self.device)

    def transpose(self, dim0, dim1):
        """
        Transpose the dimensions of the input RingTensor.

        The method uses *torch.transpose* to swap the specified dimensions (dim0 and dim1) of the RingTensor.

        :param dim0: The first dimension to swap.
        :type dim0: int
        :param dim1: The second dimension to swap.
        :type dim1: int
        :return: A new RingTensor with the transposed dimensions.
        :rtype: RingTensor
        """
        return self.__class__(self.tensor.transpose(dim0, dim1), dtype=self.dtype, device=self.device)

    def pad(self, pad, mode='constant', value=0):
        """
        Pad the input RingTensor with a specified mode and value.

        The method uses *torch.nn.functional.pad* to add padding to the RingTensor. The type of padding is determined by the mode.

        :param pad: The number of values to pad on each side of each dimension.
        :type pad: tuple
        :param mode: The padding mode (default is 'constant').
        :type mode: str
        :param value: The value to use for padding if mode is 'constant' (default is 0).
        :type value: int or float
        :return: A new RingTensor with the padded tensor.
        :rtype: RingTensor
        """
        return self.__class__(F.pad(self.tensor, pad, mode, value), dtype=self.dtype, device=self.device)

    def squeeze(self, dim=-1):
        """
        Remove dimensions of size 1 from the RingTensor.

        This method uses *torch.squeeze* to eliminate dimensions of size 1. If a specific dimension is provided,
        only that dimension is squeezed.

        :param dim: The dimension to squeeze (default is -1, which means the last dimension).
        :type dim: int
        :return: A new RingTensor with the squeezed dimensions.
        :rtype: RingTensor
        """
        return self.__class__(self.tensor.squeeze(dim), dtype=self.dtype, device=self.device)

    def unsqueeze(self, dim=-1):
        """
        Add a dimension of size 1 to the RingTensor at the specified position.

        This method uses *torch.unsqueeze* to add a new dimension at the specified index.

        :param dim: The index at which to insert the new dimension (default is -1, which means at the last position).
        :type dim: int
        :return: A new RingTensor with the added dimension.
        :rtype: RingTensor
        """
        return self.__class__(self.tensor.unsqueeze(dim), dtype=self.dtype, device=self.device)

    def size(self, dim=None):
        """
        Get the size of the RingTensor along the specified dimension.

        If dim is not provided, the entire shape of the RingTensor is returned.

        :param dim: The dimension to get the size of (default is None).
        :type dim: int or None
        :return: The size of the specified dimension or the full shape of the tensor.
        :rtype: int or tuple
        """
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    def view(self, *args):
        """
        Reshape the RingTensor to the specified dimensions.

        The method uses *torch.view* to create a new RingTensor with the desired shape. It maintains the data
        layout in memory.

        :param args: The desired shape for the output RingTensor.
        :type args: tuple[Any ...]
        :return: A new RingTensor with the specified shape.
        :rtype: RingTensor
        """
        view = self.__class__(self.tensor.view(*args), dtype=self.dtype, device=self.device)
        view.bit_len = self.bit_len
        return view

    def flatten(self, start_dim=0, end_dim=-1):
        """
        Flatten the input RingTensor from start_dim to end_dim.

        This method uses *torch.flatten* to combine the specified dimensions into a single dimension.

        :param start_dim: The first dimension to flatten (default is 0).
        :type start_dim: int
        :param end_dim: The last dimension to flatten (default is -1, which means the last dimension).
        :type end_dim: int
        :return: A new RingTensor that has been flattened.
        :rtype: RingTensor
        """
        new = self.__class__(self.tensor.flatten(start_dim, end_dim), dtype=self.dtype, device=self.device)
        new.bit_len = self.bit_len
        return new

    def permute(self, *dims):
        """
        Permute the dimensions of the input RingTensor.

        This method uses *torch.permute* to rearrange the dimensions of the RingTensor according to the specified order.

        :param dims: The desired order of dimensions.
        :type dims: tuple of int
        :return: A new RingTensor with the dimensions permuted.
        :rtype: RingTensor
        """
        new = self.__class__(self.tensor.permute(*dims), dtype=self.dtype, device=self.device)
        new.bit_len = self.bit_len
        return new

    def contiguous(self):
        """
        Return a contiguous RingTensor in memory.

        This method uses *torch.contiguous* to ensure that the RingTensor is stored in contiguous memory,
        which is often required for certain operations.

        :return: A new RingTensor that is contiguous in memory.
        :rtype: RingTensor
        """
        new = self.__class__(self.tensor.contiguous(), dtype=self.dtype, device=self.device)
        new.bit_len = self.bit_len
        return new

    def tolist(self):
        """
        Convert the RingTensor to a nested Python list.

        This method uses *torch.tolist* to transform the tensor into a standard nested list representation.

        :return: A nested list representation of the RingTensor.
        :rtype: list
        """
        return self.tensor.tolist()

    def numel(self):
        """
        Get the total number of elements in the RingTensor.

        This method uses *torch.numel* to return the total count of elements in the RingTensor.

        :return: The number of elements in the RingTensor.
        :rtype: int
        """
        return self.tensor.numel()

    def signbit(self):
        """
        Get the sign bit of the RingTensor.

        This method uses *torch.signbit* to create a RingTensor indicating the sign of each element.
        It returns a tensor with bits set for negative values.

        :return: A new RingTensor with the sign bits of the elements.
        :rtype: RingTensor
        """
        msb = torch.signbit(self.tensor) + 0
        return self.__class__(msb.to(data_type), self.dtype)

    def bit_slice(self, start, end):
        """
        Extract a slice of bits from the RingTensor.

        This method allows for bitwise slicing of the RingTensor, returning the bits from the specified start index to the end index.

        .. note::
            The start index is inclusive, but the end index is exclusive.

        :param start: The starting bit index .
        :type start: int
        :param end: The ending bit index.
        :type end: int
        :return: A new RingTensor containing the specified slice of bits.
        :rtype: RingTensor
        :raises AssertionError: If the bit indices are out of range.
        """
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
        """
        Expand the RingTensor to the specified sizes.

        This method uses *torch.expand* to create a new RingTensor with the specified sizes,
        broadcasting the original RingTensor's values as necessary.

        :param sizes: The desired sizes for each dimension.
        :type sizes: tuple of int
        :return: A new RingTensor with the expanded size.
        :rtype: RingTensor
        """
        return self.__class__(self.tensor.expand(*sizes), dtype=self.dtype, device=self.device)

    def argsort(self, dim=-1):
        """
        Sort the elements of the RingTensor along a specified dimension and return the indices of the sorted elements.

        :param dim: The dimension along which to sort the elements (default is -1, which means the last dimension).
        :return:
        """
        return self.__class__(torch.argsort(self.tensor, dim=dim), dtype=self.dtype, device=self.device,
                              bit_len=self.bit_len)

    def index_add_(self, dim, index, source):
        """
        Add values from the source RingTensor to the specified indices in the current RingTensor.

        This method uses *torch.index_add_* to perform an in-place addition of values from the source RingTensor
        at the specified indices along the given dimension.

        :param dim: The dimension along which to index.
        :type dim: int
        :param index: The indices at which to add values.
        :type index: RingTensor or torch.Tensor
        :param source: The RingTensor containing values to add.
        :type source: RingTensor
        :return: The updated RingTensor after in-place addition.
        :rtype: RingTensor
        """
        if isinstance(index, RingTensor):
            index = index.tensor
        self.tensor.index_add_(dim, index, source.tensor)
        return self

    @staticmethod
    def where(condition, x, y):
        """
        Select elements from x or y based on a condition RingTensor.

        This method returns a RingTensor where elements are taken from x if the condition is true,
        and from y if false. Supports both RingTensor and int types for x and y.

        :param condition: A RingTensor representing the condition.
        :type condition: RingTensor
        :param x: The RingTensor or scalar to select from when condition is true.
        :type x: RingTensor or int
        :param y: The tensor or scalar to select from when condition is false.
        :type y: RingTensor or int
        :return: A new RingTensor with selected values based on the condition.
        :rtype: RingTensor
        """
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
        """
        Generate a RingTensor filled with random integers.

        This method creates a RingTensor with random integers between down_bound and upper_bound
        with the specified shape.

        :param shape: The shape of the output RingTensor.
        :type shape: tuple[int] or list[int]
        :param dtype: The data type of the RingTensor (default is 'int').
        :type dtype: str
        :param device: The device to allocate the RingTensor on (default is DEVICE).
        :type device: str
        :param down_bound: The lower bound for random integers (inclusive).
        :type down_bound: int
        :param upper_bound: The upper bound for random integers (exclusive).
        :type upper_bound: int
        :return: A new RingTensor filled with random integers.
        :rtype: RingTensor
        """
        v = torch.randint(down_bound, upper_bound, shape, dtype=data_type, device=device)
        return RingTensor(v, dtype, device)

    @staticmethod
    def empty(size, dtype='int', device=DEVICE):
        """
        Create an uninitialized RingTensor of specified size.

        This method returns a RingTensor without initializing its values.

        :param size: The size of the output RingTensor.
        :type size: tuple[int] or list[int]
        :param dtype: The data type of the tensor (default is 'int').
        :type dtype: str
        :param device: The device to allocate the tensor on (default is DEVICE).
        :type device: str
        :return: A new uninitialized RingTensor.
        :rtype: RingTensor
        """
        return RingTensor(torch.empty(size, dtype=data_type), dtype, device)

    @staticmethod
    def empty_like(tensor):
        """
        Create an uninitialized RingTensor with the same shape as the input RingTensor.

        This method returns a RingTensor with the same shape as the provided RingTensor but without
        initializing its values.

        .. note::
            The main difference between the :meth:`empty` and :meth:`empty_like` methods is:
                * *empty* creates a new RingTensor with a specified shape and data type.
                * *empty_like* creates a new RingTensor with the same shape as the given tensor.

        :param tensor: The tensor to mimic.
        :type tensor: RingTensor
        :return: A new uninitialized RingTensor with the same shape.
        :rtype: RingTensor
        :raises TypeError: If the input tensor is not a RingTensor.
        """
        if isinstance(tensor, RingTensor):
            return RingTensor(torch.empty_like(tensor.tensor), tensor.dtype, tensor.device)
        else:
            raise TypeError("unsupported operand type(s) for empty_like 'RingTensor' and ", type(tensor))

    @staticmethod
    def zeros(size, dtype='int', device=DEVICE):
        """
        Create a RingTensor filled with zeros.

        This method returns a RingTensor initialized to zero with the specified size.

        :param size: The size of the output RingTensor.
        :type size: tuple[int] or list[int]
        :param dtype: The data type of the tensor (default is 'int').
        :type dtype: str
        :param device: The device to allocate the tensor on (default is DEVICE).
        :type device: str
        :return: A new RingTensor filled with zeros.
        :rtype: RingTensor
        """
        return RingTensor(torch.zeros(size, dtype=data_type, device=device), dtype)

    @staticmethod
    def zeros_like(tensor, dtype='int', device=DEVICE):
        """
        Create a RingTensor of zeros with the same shape as the input RingTensor.

        This method returns a RingTensor filled with zeros that has the same shape as the provided RingTensor.

        .. note::
            The difference between :meth:`zeros` and :meth:`zeros_like` is as :meth:`empty` and :meth:`empty_like`

        :param tensor: The tensor to mimic.
        :type tensor: RingTensor or torch.Tensor
        :param dtype: The data type of the tensor (default is 'int').
        :type dtype: str
        :param device: The device to allocate the tensor on (default is DEVICE).
        :type device: str
        :return: A new RingTensor filled with zeros.
        :rtype: RingTensor
        :raises TypeError: If the input tensor is neither a RingTensor nor a torch.Tensor.
        """
        if isinstance(tensor, RingTensor):
            return RingTensor(torch.zeros_like(tensor.tensor), tensor.dtype, tensor.device)
        elif isinstance(tensor, torch.Tensor):
            return RingTensor(torch.zeros_like(tensor), dtype, device)
        else:
            raise TypeError("unsupported operand type(s) for zeros_like 'RingTensor' and ", type(tensor))

    @staticmethod
    def ones(size, dtype='int', device=DEVICE):
        """
        Create a RingTensor filled with ones.

        This method returns a RingTensor initialized to one with the specified size.

        :param size: The size of the output RingTensor.
        :type size: tuple of int
        :param dtype: The data type of the RingTensor (default is 'int').
        :type dtype: str or dtype
        :param device: The device to allocate the RingTensor on (default is DEVICE).
        :type device: str
        :return: A new RingTensor filled with ones.
        :rtype: RingTensor
        """
        scale = DTYPE_SCALE_MAPPING[dtype]
        return RingTensor(torch.ones(size, dtype=data_type, device=device) * scale, dtype)

    @staticmethod
    def ones_like(tensor, dtype='int', device=DEVICE):
        """
        Create a RingTensor of ones with the same shape as the input tensor.

        This method returns a RingTensor filled with ones that has the same shape as the provided tensor.

        .. note::
            The difference between :meth:`ones` and :meth:`ones_like` is as :meth:`empty` and :meth:`empty_like`

        :param tensor: The tensor to mimic.
        :type tensor: RingTensor or torch.Tensor
        :param dtype: The data type of the tensor (default is 'int').
        :type dtype: str
        :param device: The device to allocate the tensor on (default is DEVICE).
        :type device: str
        :return: A new RingTensor filled with ones.
        :rtype: RingTensor
        :raises TypeError: If the input tensor is neither a RingTensor nor a torch.Tensor.
        """
        if isinstance(tensor, RingTensor):
            return RingTensor(torch.ones_like(tensor.tensor) * tensor.scale, tensor.dtype, tensor.device)
        elif isinstance(tensor, torch.Tensor):
            scale = DTYPE_SCALE_MAPPING[dtype]
            return RingTensor(torch.ones_like(tensor) * scale, dtype, device)
        else:
            raise TypeError("unsupported operand type(s) for ones_like 'RingTensor' and ", type(tensor))

    @staticmethod
    def full(size, fill_value, device=DEVICE):
        """
        Create a RingTensor filled with a specified value.

        This method initializes a RingTensor of the specified size with the given fill value.

        :param size: The size of the output RingTensor.
        :type size: tuple of int
        :param fill_value: The value to fill the RingTensor with.
        :type fill_value: Union
        :param device: The device to allocate the RingTensor on (default is DEVICE).
        :type device: str
        :return: A new RingTensor filled with the specified value.
        :rtype: RingTensor
        """
        return RingTensor.convert_to_ring(torch.full(size, fill_value, device=device))

    @staticmethod
    def full_like(tensor, fill_value, device=DEVICE):
        """
        Create a RingTensor filled with a specified value with the same shape as the input tensor.

        This method initializes a RingTensor with the same shape as the provided tensor,
        filled with the given value.

        :param tensor: The tensor to mimic.
        :type tensor: RingTensor or torch.Tensor
        :param fill_value: The value to fill the RingTensor with.
        :type fill_value: Union
        :param device: The device to allocate the RingTensor on (default is DEVICE).
        :type device: str
        :return: A new RingTensor filled with the specified value.
        :rtype: RingTensor
        :raises TypeError: If the input tensor is neither a RingTensor nor a torch.Tensor.
        """
        if isinstance(tensor, RingTensor):
            return RingTensor.full(tensor.shape, fill_value, tensor.device)
        elif isinstance(tensor, torch.Tensor):
            return RingTensor.convert_to_ring(torch.full_like(tensor, fill_value, device=device))
        else:
            raise TypeError("unsupported operand type(s) for full_like 'RingTensor' and ", type(tensor))

    @staticmethod
    def arange(start, end, step=1, dtype='int', device=DEVICE):
        """
        Create a RingTensor containing a sequence of values.

        This method generates a RingTensor with values ranging from start to end,
        spaced by the given *step*.

        :param start: The starting value of the sequence.
        :type start: int
        :param end: The end value of the sequence (exclusive).
        :type end: int
        :param step: The spacing between consecutive values (default is 1).
        :type step: int
        :param dtype: The data type of the output RingTensor.
        :type dtype: str
        :param device: The device to allocate the RingTensor on (default is DEVICE).
        :type device: str
        :return: A new RingTensor containing the generated sequence.
        :rtype: RingTensor
        """
        return RingTensor(torch.arange(start, end, step, dtype=data_type, device=device), dtype)

    @staticmethod
    def batch_randperm(num, n, device=DEVICE):
        """
        Generate a batch of random permutations of integers from 0 to n-1.
        :param num:
        :param n:
        :param device:
        :return:
        """
        rand = torch.rand(num, n, device=device)
        return RingTensor(rand.argsort(dim=1), dtype='int', device=device)
