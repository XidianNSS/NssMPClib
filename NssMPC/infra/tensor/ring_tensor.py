#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.
from __future__ import annotations

import math
from functools import singledispatchmethod
from os import PathLike

import torch
import torch.nn.functional as F

from NssMPC.config import BIT_LEN, DEVICE, data_type, DTYPE_MAPPING, DTYPE_SCALE_MAPPING, HALF_RING
from NssMPC.infra.utils.cuda_utils import cuda_matmul, cuda_rotate


class RingTensor(object):
    # TODO 运算符重载在不支持类型时应 return NotImplemented 否则不会去检查右值是否支持了此运算
    # TODO 支持可变bit_len?
    """
    Define the tensor on the ring. Support for the part of operations on pytorch tensor.

    Attributes:
        tensor (torch.Tensor): The pytorch tensor.
        dtype (str): The dtype of the RingTensor.
        bit_len (int): The expected length of the binary representation after conversion.
    """

    @singledispatchmethod
    def __init__(self):
        """
        Initialize method.

        Using `@singledispatchmethod` allows you to select different initialization logic depending on the input type.
        """
        self.bit_len = None
        self.dtype = None
        self.tensor = None

    @__init__.register(torch.Tensor)
    def from_tensor(self, tensor, dtype='int', device=DEVICE, bit_len=BIT_LEN):
        """
        Initialize RingTensor from a torch.Tensor.

        Args:
            tensor (torch.Tensor): Input tensor.
            dtype (str): Data type ('int' or 'float').
            device (str): Device to store the tensor.
            bit_len (int): Bit length.

        Examples:
            >>> RingTensor(torch.tensor([1, 2, 3]))
        """
        self.tensor = tensor.to(device) if tensor.device.type != device else tensor
        self.dtype = dtype
        self.bit_len = bit_len

    @__init__.register(int)
    @__init__.register(list)
    def from_item(self, item, dtype='int', device=DEVICE, bit_len=BIT_LEN):
        """
        Initialize RingTensor from an item (int or list).

        Args:
            item (int or list): Input item.
            dtype (str): Data type ('int' or 'float').
            device (str): Device to store the tensor.
            bit_len (int): Bit length.

        Examples:
            >>> RingTensor([1, 2, 3])
        """
        self.tensor = torch.tensor(item, dtype=data_type, device=device)
        self.dtype = dtype
        self.bit_len = bit_len

    @property
    def scale(self):
        """
        Return the scale of the RingTensor.

        Obtain the corresponding value from `DTYPE_SCALE_MAPPING` based on dtype.

        Returns:
            int: The scale value.

        Examples:
            >>> ring_tensor.scale
        """
        return DTYPE_SCALE_MAPPING[self.dtype]

    @property
    def device(self):
        """
        Return the device type of the current RingTensor.

        Returns:
            str: The device type expressed as a string.

        Examples:
            >>> ring_tensor.device
        """
        return self.tensor.device.__str__()

    @property
    def shape(self):
        """
        Return RingTensor shape information.

        Returns:
            torch.Size: The shape of the tensor.

        Examples:
            >>> ring_tensor.shape
        """
        return self.tensor.shape

    @property
    def T(self):
        """
        Create a new RingTensor instance with the old dtype, device, and bit_len.

        Returns:
            RingTensor: Transpose of the current RingTensor.

        Examples:
            >>> ring_tensor.T
        """
        new_value = self.tensor.T
        return self.__class__(new_value, self.dtype, self.device, self.bit_len)

    def __str__(self):
        """
        A custom string representation that provides class name, RingTensor value, data type, and scaling information.

        Returns:
            str: A string that represents the RingTensor instance.

        Examples:
            >>> print(ring_tensor)
        """
        return f"{self.__class__.__name__}\n value:{self.tensor} \n dtype:{self.dtype} \n scale:{self.scale}"

    def __getitem__(self, item):
        """
        Support index access.

        Args:
            item (int or slice or list or tuple): The index.

        Returns:
            RingTensor: A new RingTensor instance containing the indexed part.

        Examples:
            >>> ring_tensor[0]
        """
        return self.__class__(self.tensor[item], self.dtype, self.device, self.bit_len)

    def __setitem__(self, key, value):
        """
        Support index assignment.

        Args:
            key (int or slice or list or tuple): The index.
            value (RingTensor): The value corresponding to the index.

        Raises:
            TypeError: The value passed in is not of the RingTensor type.

        Examples:
            >>> ring_tensor[0] = other_ring_tensor
        """
        if isinstance(value, RingTensor):
            self.tensor[key] = value.tensor.clone()
        else:
            raise TypeError(f"unsupported operand type(s) for setitem '{type(self)}' and {type(value)}")

    def __len__(self):
        """
        Return the length of the current RingTensor.

        Returns:
            int: The length of the tensor.

        Examples:
            >>> len(ring_tensor)
        """
        return len(self.tensor)

    def __invert__(self):
        """
        Implement the reverse operation by bit.

        Returns:
            RingTensor: A new RingTensor representing the bitwise inversion.

        Examples:
            >>> ~ring_tensor
        """
        return self.__class__(~self.tensor, self.dtype, self.device, self.bit_len)

    def __add__(self, other):
        """
        Adds a RingTensor object with a corresponding type.

        The type of other is determined first:
            * If `other` is `RingTensor`: perform RingTensor addition and update bit_len to the maximum of the two.
            * If `other` is `torch.Tensor` or `int`: add directly.

        Args:
            other (RingTensor or torch.Tensor or int): The object to be added to self.

        Returns:
            RingTensor: The result of the addition.

        Raises:
            TypeError: If `other` is not a supported type.

        Examples:
            >>> ring_tensor + other_ring_tensor
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

        Args:
            other (RingTensor or torch.Tensor or int): The object to be added.

        Returns:
            RingTensor: The result of the addition.

        Examples:
            >>> 1 + ring_tensor
        """
        return self.__add__(other)

    def __iadd__(self, other):
        """
        In situ addition operation, using the `+=` operator.

        Similar to `__add__`, but directly modifies the value of `self.tensor`. Also check the type and process, update `self.tensor`.

        Args:
            other (RingTensor or torch.Tensor or int): The object to be added.

        Returns:
            RingTensor: Self.

        Examples:
            >>> ring_tensor += other
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
            * If `other` is `RingTensor`: perform RingTensor subtraction and update bit_len to the maximum of the two.
            * If `other` is `torch.Tensor` or `int`: subtract directly.

        Args:
            other (RingTensor or torch.Tensor or int): Subtrahend.

        Returns:
            RingTensor: The result of the subtraction.

        Examples:
            >>> ring_tensor - other
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

        Args:
            other (RingTensor or torch.Tensor or int): Minuend.

        Returns:
            RingTensor: The result of the subtraction.

        Examples:
            >>> 1 - ring_tensor
        """
        return -self.__sub__(other)

    def __isub__(self, other):
        """
        Perform a subtraction operation in place, using the `-=` operator.

        Args:
            other (RingTensor or torch.Tensor or int): Subtrahend.

        Returns:
            RingTensor: Self.

        Examples:
            >>> ring_tensor -= other
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
            * If `other` is `RingTensor`: Calculate the new value based on the scale property of both objects and select the appropriate dtype.
            * If `other` is `int`: multiply directly.
            * If `other` is not a supported type: return `NotImplemented`.

        Args:
            other (RingTensor or int): Multiplier.

        Returns:
            RingTensor: Multiplication result.

        Examples:
            >>> ring_tensor * other
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
        Right-hand multiplication operation.

        Args:
            other (RingTensor or int): Multiplier.

        Returns:
            RingTensor: Multiplication result.

        Examples:
            >>> 2 * ring_tensor
        """
        return self.__mul__(other)

    def __imul__(self, other):
        """
        Local multiplication operation, using the `*=` operator.

        Args:
            other (RingTensor or int): Multiplier.

        Returns:
            RingTensor: Self.

        Examples:
            >>> ring_tensor *= other
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
        Implement modular operation `%`.

        * If `other` is an integer, 'self.tensor % other' is evaluated, then calculate the new `bit_len`.
            * If `self.bit_len` is greater than the logarithm of `other`, its pair value is used.
            * otherwise `self.bit_len` is used.

        Args:
            other (int): Modulo.

        Returns:
            RingTensor: The result after taking the mold.

        Raises:
            TypeError: If `other` is not an integer.

        Examples:
            >>> ring_tensor % 2
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
        Implement matrix multiplication `@`.

        The type of `other` is determined first, if the `other` is not RingTensor, raise `TypeError`. When `other`
        is a RingTensor, First, make sure `self` and `other` have the same data type. If not, an assertion error is
        thrown, then check whether the device is a CUDA device:
            * If on a CUDA device, call the cuda_matmul function for matrix multiplication.
            * If the device is not CUDA, the matrix multiplication of the CPU is performed.

        Args:
            other (RingTensor): Multiplier.

        Returns:
            RingTensor: The result of matrix multiplication.

        Examples:
            >>> ring_tensor @ other_ring_tensor
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
        Implement true division `/`.

        The type of other is determined first:
            * If `other` is `RingTensor`: Then check whether the scale is the same and perform the calculation.
            * If `other` is `torch.Tensor` or `int`: divide directly.

        Note:
            Use torch.round() to process the final value, ensuring that the result returned is an integer.

        Args:
            other (RingTensor or torch.Tensor or int): Divisor.

        Returns:
            RingTensor: Quotient.

        Raises:
            TypeError: If `other` is not a supported type.

        Examples:
            >>> ring_tensor / other
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
        Implement proper division in place `/=`.

        Similar to `__truediv__`, but directly modifies the value of `self.tensor`. Also check the type and process, update `self.tensor`.

        Args:
            other (RingTensor or torch.Tensor or int): Divisor.

        Returns:
            RingTensor: Self.

        Examples:
            >>> ring_tensor /= other
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
        Implement floor division `//`.

        The processing is similar to `__truediv__`, but uses `//` to perform floor division.

        Args:
            other (RingTensor or torch.Tensor or int): Divisor.

        Returns:
            RingTensor: Quotient.

        Raises:
            TypeError: If `other` is not a supported type.

        Examples:
            >>> ring_tensor // other
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
        Implement local floor division operations `//=`.

        Similar to `__floordiv__`, but modifying `self.tensor` directly, perform floor division and update the value.

        Args:
            other (RingTensor or torch.Tensor or int): Divisor.

        Returns:
            RingTensor: Self.

        Examples:
            >>> ring_tensor //= other
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
        Implement the unary minus operation `-`.

        Take the negative of each element in `self.tensor`, at the shame time, keep the same data types and devices.

        Returns:
            RingTensor: The negative number of each element.

        Examples:
            >>> -ring_tensor
        """
        new_value = -self.tensor
        return self.__class__(new_value, dtype=self.dtype, device=self.device)

    def __eq__(self, other):
        """
        Implement equal operation `==`.

        The type of other is determined first:
            * If `other` is `RingTensor`: Compare two tensors to see if they are equal, and the result is stored in new_value.
            * If `other` is `int`: Multiply this by `self.scale` for comparison.

        Args:
            other (RingTensor or int): The object to compare with `self`.

        Returns:
            RingTensor: Boolean comparison results.

        Raises:
            TypeError: If `other` is not a supported type.

        Examples:
            >>> ring_tensor == other
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
        Implementation is not equal to operation `!=`.

        Similar to the `__eq__` method, compare the inequality of two tensors.

        Args:
            other (RingTensor or int): The object to compare with `self`.

        Returns:
            RingTensor: Boolean comparison results.

        Raises:
            TypeError: If `other` is not a supported type.

        Examples:
            >>> ring_tensor != other
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
        Implement greater-than operation `>`.

        The type of other is determined first:
            * If `other` is `RingTensor`: Compares the elements of the two tensors and generates the Boolean array new_value.
            * If `other` is `int`: Multiply this by `self.scale` for comparison.

        Args:
            other (RingTensor or int): The object to compare with `self`.

        Returns:
            RingTensor: Boolean comparison results.

        Raises:
            TypeError: If `other` is not a supported type.

        Examples:
            >>> ring_tensor > other
        """
        if isinstance(other, RingTensor):
            new_value = (self.tensor > other.tensor)
        elif isinstance(other, int):
            new_value = (self.tensor > other * self.scale)
        else:
            raise TypeError(
                "unsupported operand type(s) for > 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value.to(self.tensor.dtype), self.dtype, self.device)

    def __ge__(self, other):
        """
        Implement greater-than-equal operations `>=`.

        Similar to `__gt__`, but with greater than or equal comparison symbols.

        Args:
            other (RingTensor or int): The object to compare with `self`.

        Returns:
            RingTensor: Boolean comparison results.

        Raises:
            TypeError: If `other` is not a supported type.

        Examples:
            >>> ring_tensor >= other
        """
        if isinstance(other, RingTensor):
            new_value = (self.tensor >= other.tensor)
        elif isinstance(other, int):
            new_value = (self.tensor >= other * self.scale)
        else:
            raise TypeError(
                "unsupported operand type(s) for >= 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value.to(self.tensor.dtype), self.dtype, self.device)

    def __lt__(self, other):
        """
        Implement the less than operation `<`.

        Similar to `__gt__`, the resulting Boolean array new_value reflects less than the result of the comparison.

        Args:
            other (RingTensor or int): The object to compare with `self`.

        Returns:
            RingTensor: Boolean comparison results.

        Raises:
            TypeError: If `other` is not a supported type.

        Examples:
            >>> ring_tensor < other
        """
        if isinstance(other, RingTensor):
            new_value = (self.tensor < other.tensor)
        elif isinstance(other, int):
            new_value = (self.tensor < other * self.scale)
        else:
            raise TypeError(
                "unsupported operand type(s) for < 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value.to(self.tensor.dtype), self.dtype, self.device)

    def __le__(self, other):
        """
        Implements the less than or equal operation `<=`.

        Similar to `__ge__`, the type is judged and a less than or equal comparison is made.

        Args:
            other (RingTensor or int): The object to compare with `self`.

        Returns:
            RingTensor: Boolean comparison results.

        Raises:
            TypeError: If `other` is not a supported type.

        Examples:
            >>> ring_tensor <= other
        """
        if isinstance(other, RingTensor):
            new_value = (self.tensor <= other.tensor)
        elif isinstance(other, int):
            new_value = (self.tensor <= other * self.scale)
        else:
            raise TypeError(
                "unsupported operand type(s) for <= 'RingTensor' and ", type(other), 'please convert to '
                                                                                     'ring first')
        return self.__class__(new_value.to(self.tensor.dtype), self.dtype, self.device)

    def __xor__(self, other):
        """
        Implement bitwise XOR operations `^`.

        The type of other is determined first:
            * If `other` is `RingTensor`: Bitwise xOR operations using tensors.
            * If `other` is `torch.Tensor` or `int`: Also do XOR by bit.

        Args:
            other (RingTensor or int or torch.Tensor): Object that makes a bit-by-bit xOR with `self`.

        Returns:
            RingTensor: Boolean comparison results.

        Raises:
            TypeError: If `other` is not a supported type.

        Examples:
            >>> ring_tensor ^ other
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
        Implement bitwise or operations `|`.

        The type of other is determined first:
            * If `other` is `RingTensor`: Perform bitwise or operations.
            * If `other` is `int`: Perform the corresponding bitwise or operation.

        Args:
            other (RingTensor or int): Object that makes a bitwise or with `self`.

        Returns:
            RingTensor: Boolean comparison results.

        Raises:
            TypeError: If `other` is not a supported type.

        Examples:
            >>> ring_tensor | other
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
        Implement bit-and-operate `&`.

        The type of other is determined first:
            * If `other` is `RingTensor`: Perform bitwise and operations.
            * If `other` is `int`: Execute by bit and.

        Args:
            other (RingTensor or int): Object that makes a bitwise and with `self`.

        Returns:
            RingTensor: Boolean comparison results.

        Raises:
            TypeError: If `other` is not a supported type.

        Examples:
            >>> ring_tensor & other
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
        Right shift operation `>>`.

        The type of other is determined first:
            * If `other` is `RingTensor`: Move the value of the `self` object to the right by `other` bits.
            * If `other` is `int`: Also perform the right shift.

        Args:
            other (RingTensor or int): Number of bits shifted to the right.

        Returns:
            RingTensor: The result of the `self` shift.

        Raises:
            TypeError: If `other` is not a supported type.

        Examples:
            >>> ring_tensor >> 1
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
        Implement left shift operation `<<`.

        Similar to `__rshift__`, but the operation is to move left.

        Args:
            other (RingTensor or int): Number of bits shifted to the left.

        Returns:
            RingTensor: The result of the `self` shift.

        Raises:
            TypeError: If `other` is not a supported type.

        Examples:
            >>> ring_tensor << 1
        """
        if isinstance(other, RingTensor):
            return self.__class__(self.tensor << other.tensor, self.dtype, self.device)
        elif isinstance(other, int):
            return self.__class__(self.tensor << other, self.dtype)
        else:
            raise TypeError(
                "unsupported operand type(s) for << 'RingTensor' and ", type(other), 'please convert to ring first')

    @classmethod
    def convert_to_ring(cls, item: int | float | list | torch.Tensor) -> RingTensor:
        """
        Converts the item to RingTensor.

        If the `item` is an integer, a floating point number, or a list, use `torch.tensor` to convert it to a
        PyTorch tensor. Then assert whether the converted item is of type torch.Tensor. Get the data type scaling
        factor scale for item.
            * If scale is not equal to 1, round the item and multiply it by scale, otherwise leave it unchanged.
            * Set dtype to `int` or `float` depending on the value of scale.

        Args:
            item: item to convert.

        Returns:
            RingTensor: A RingTensor object converted from item.

        Examples:
            >>> RingTensor.convert_to_ring(1.5)
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
    def load_from_file(cls, file_path: PathLike) -> torch.Tensor:
        # TODO: 返回的是tensor，而非RingTensor;改为RingTensor而类型无法确定。
        """
        Loads a RingTensor from a file.

        Args:
            file_path: The path of the loaded file.

        Returns:
            torch.Tensor: The contents stored in the file.

        Examples:
            >>> RingTensor.load_from_file('tensor.pt')
        """
        return torch.load(file_path)

    @classmethod
    def mul(cls, x: RingTensor, y: RingTensor) -> RingTensor:
        """
        Multiplies two RingTensors without truncation.

        This only applies if `x` and `y` are both RingTensor.

        Args:
            x: Multiplicand.
            y: Multiplier.

        Returns:
            RingTensor: Multiplication result.

        Examples:
            >>> RingTensor.mul(x, y)
        """
        return cls(x.tensor * y.tensor, x.dtype)

    @classmethod
    def matmul(cls, x: RingTensor, y: RingTensor) -> RingTensor | None:
        """
        Matrix Multiplies two RingTensors without truncation.

        First, check the device type of x and y:
            * If the device is a CPU, use torch.matmul to compute matrix multiplication.
            * If the device is a CUDA, call the cuda_matmul function for matrix multiplication.

        Args:
            x: Multiplicand.
            y: Multiplier.

        Returns:
            RingTensor: The result of matrix multiplication.

        Raises:
            TypeError: x and y are not on the same device.

        Examples:
            >>> RingTensor.matmul(x, y)
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
    def cat(cls, tensor_list: list[RingTensor], dim: int = 0) -> RingTensor:
        """
        Concatenate multiple RingTensor objects in the specified dimension.

        Note:
            All RingTensor to be spliced must have the same shape in dimensions other than the spliced dimension. If this condition is not met, the method throws an error.

        Args:
            tensor_list: A list of RingTensor to splice.
            dim: Concatenated dimensions (default is 0).

        Returns:
            RingTensor: The result after splicing.

        Examples:
            >>> RingTensor.cat([t1, t2], dim=0)
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
    def stack(cls, tensor_list: list[RingTensor], dim: int = 0) -> RingTensor:
        """
        Stack multiple RingTensor objects on a specified dimension.

        First, Use assert to ensure that all RingTensor data types in `tensor_list` are the same. If not,
        throw an error. Then use torch.stack to stack tensor_list tensors in the specified dimension.

        Args:
            tensor_list: A list of RingTensor to splice.
            dim: Stack dimensions (default is 0).

        Returns:
            RingTensor: The result after stacking.

        Examples:
            >>> RingTensor.stack([t1, t2], dim=0)
        """
        assert all([t.dtype == tensor_list[0].dtype for t in tensor_list]), "The element type should be the same"
        return cls(torch.stack([t.tensor for t in tensor_list], dim), tensor_list[0].dtype, tensor_list[0].device,
                   tensor_list[0].bit_len)

    @classmethod
    def gather(cls, input: RingTensor, dim: int, index: RingTensor | torch.Tensor) -> RingTensor:
        """
        Collect data from the input RingTensor according to the index.

        Use `torch.gather` to collect data from `input.tensor` on the specified dimension to form a new RingTensor.

        Args:
            input: RingTensor to operate.
            dim: Dimensions to collect.
            index: Index tensor.

        Returns:
            RingTensor: Results collected according to the index.

        Examples:
            >>> RingTensor.gather(input, 0, index)
        """
        if isinstance(index, RingTensor):
            index = index.tensor
        return cls(torch.gather(input.tensor, dim, index), input.dtype, input.device, input.bit_len)

    @classmethod
    def roll(cls, input, shifts, dims=0):
        """
        Scroll the input RingTensor.

        Roll shift using `torch.roll`.

        Args:
            input (RingTensor): RingTensor object to be operated.
            shifts (int): The amount of displacement of rolling.
            dims (int): Dimensions to roll.

        Returns:
            RingTensor: `input` after rolling shift.

        Examples:
            >>> RingTensor.roll(input, 1, 0)
        """
        return cls(torch.roll(input.tensor, shifts=shifts, dims=dims), input.dtype, input.device, input.bit_len)

    @classmethod
    def rotate(cls, input, shifts):
        """
        For a two-dimensional RingTensor of input, shift each row to the left or right depending on the corresponding element in shifts.

        Ensure that the shifts data type is a tensor, If shifts is a list or RingTensor, Convert it to `torch.tensor`.
            * if shifts > 0, rotate to the right
            * if shifts < 0, rotate to the left

        Args:
            input (RingTensor): RingTensor object to be operated.
            shifts (RingTensor or list or torch.Tensor): Rotational displacement.

        Returns:
            RingTensor: Input after rotation operation.

        Examples:
            >>> RingTensor.rotate(input, shifts)
        """
        if isinstance(shifts, list):
            shifts = torch.tensor(shifts, dtype=data_type, device=input.device)
        elif isinstance(shifts, RingTensor):
            shifts = shifts.tensor

        if 'cuda' in input.device:
            result = cuda_rotate(input.tensor, shifts)
            return cls(result, input.dtype, input.device, input.bit_len)
        else:
            n = input.shape[1]
            rows = torch.arange(input.shape[0]).view(-1, 1)
            indices = (torch.arange(n, device=shifts.device) - shifts.view(-1, 1)) % n
            result = input[rows, indices]

            return result

    @classmethod
    def onehot(cls, input, num_classes=-1):
        """
        Creates a one-hot RingTensor from input RingTensor.

        Use `torch.nn.functional.one_hot` for a heat encoding and multiply by the input scaling factor.

        Args:
            input (RingTensor): The indices.
            num_classes (int): The number of classes.

        Returns:
            RingTensor: A one-hot RingTensor.

        Examples:
            >>> RingTensor.onehot(input, num_classes=10)
        """
        return cls(F.one_hot(input.tensor.to(torch.int64), num_classes).to(data_type) * input.scale, input.dtype,
                   input.device)

    @classmethod
    def exp(cls, x):
        """
        Calculate the exponent of the input RingTensor.

        Calculate using `torch.exp` after converting x to the real number field. The exponent based on e is (e^x).

        Args:
            x (RingTensor): RingTensor to be calculated.

        Returns:
            RingTensor: Input index.

        Examples:
            >>> RingTensor.exp(x)
        """
        assert x.dtype == 'float', "The input tensor should be float type"
        return cls.convert_to_ring(torch.exp(x.convert_to_real_field()))

    @classmethod
    def exp2(cls, x):
        """
        Compute the base 2 exponent of the input RingTensor. The exponent with base 2 is (2^x).

        This method is similar to `exp`, but `exp2` is an exponent with base 2.

        Args:
            x (RingTensor): RingTensor to be calculated.

        Returns:
            RingTensor: Input index.

        Examples:
            >>> RingTensor.exp2(x)
        """
        return cls.convert_to_ring(torch.exp2(x.convert_to_real_field()))

    @classmethod
    def tanh(cls, x):
        """
        Calculate the double of the input RingTensor x.

        First, use the `convert_to_real_field()` function to convert x from a ring domain to the real number domain.

        Note:
            The function returns values within the range of [-1, 1].

        Args:
            x (RingTensor): RingTensor to be calculated.

        Returns:
            RingTensor: Calculation result.

        Examples:
            >>> RingTensor.tanh(x)
        """
        return cls.convert_to_ring(torch.tanh(x.convert_to_real_field()))

    def convert_to_real_field(self):
        """
        Convert the RingTensor from ring field to real field.

        RingTensor types mapped to PyTorch depending on the bit length (`BIT_LEN`) and data type (`dtype`) to get `torch_type`.

        Returns:
            torch.Tensor: The torch.Tensor with numerical values of the real number field.

        Examples:
            >>> ring_tensor.convert_to_real_field()
        """
        torch_type = {64: {'int': torch.int64, 'float': torch.float64},
                      32: {'int': torch.int32, 'float': torch.float32}}
        return (self.tensor / self.scale).to(torch_type[BIT_LEN][self.dtype])

    def convert_to_range(self, bit_len):
        """
        Convert the RingTensor to the ring of a specified bit length.

        Args:
            bit_len (int): The specified bit length.

        Returns:
            RingTensor: The RingTensor after conversion.

        Examples:
            >>> ring_tensor.convert_to_range(32)
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

        Use `torch.sum` to calculate the sum of `self.tensor` along the specified dimension dim, while retaining the RingTensor's original data type.

        Args:
            dim (int): Specified Dimension.

        Returns:
            RingTensor: Sum result.

        Examples:
            >>> ring_tensor.sum(dim=0)
        """
        new_value = torch.sum(self.tensor, dim=dim, dtype=self.tensor.dtype)
        return self.__class__(new_value, dtype=self.dtype, device=self.device, bit_len=self.bit_len)

    def all(self):
        """
        Check whether all elements of the RingTensor are true.

        Returns:
            bool: Check the bool value of the result.

        Examples:
            >>> ring_tensor.all()
        """
        return self.tensor.all()

    def any(self):
        """
        Check if the RingTensor has at least one element that is true.

        Returns:
            bool: Check the bool value of the result.

        Examples:
            >>> ring_tensor.any()
        """
        return self.tensor.any()

    def to(self, target):
        """
        Move the RingTensor to a specified device or convert the RingTensor to a specified type.

        Args:
            target (str or dtype): The specified computing device or dtype.

        Returns:
            RingTensor: The RingTensor after the move or convert.

        Examples:
            >>> ring_tensor.to('cuda')
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

        Returns:
            RingTensor: The RingTensor after the move.

        Examples:
            >>> ring_tensor.cpu()
        """
        self.tensor = self.tensor.cpu()
        return self

    def cuda(self, device=None):
        """
        Move the RingTensor to the GPU device. If device is specified, the device is used.

        Args:
            device (str): The specified computing device.

        Returns:
            RingTensor: The RingTensor after the move.

        Examples:
            >>> ring_tensor.cuda()
        """
        self.tensor = self.tensor.cuda(device)
        return self

    def save(self, file_path):
        """
        Saves the current RingTensor instance to the specified file path.

        Serialize the self object and save it to a file using `torch.save`.

        Args:
            file_path (str): The path of the saved file.

        Examples:
            >>> ring_tensor.save('tensor.pt')
        """
        torch.save(self, file_path)

    def clone(self):
        """
        Creates a clone of the current RingTensor.

        Independent manipulation or modification in cases where it is necessary to keep the original RingTensor unchanged.

        Returns:
            RingTensor: New clone.

        Examples:
            >>> ring_tensor.clone()
        """
        clone = self.__class__(self.tensor.clone(), dtype=self.dtype, device=self.device)
        clone.bit_len = self.bit_len
        return clone

    def get_bit(self, item):
        """
        Retrieves a bit from the specified location.

        Args:
            item (int): The index of the bit to be extracted.

        Returns:
            RingTensor: The extracted bit.

        Examples:
            >>> ring_tensor.get_bit(0)
        """
        assert (self.bit_len >> item >= 0), "bit index out of range"
        return RingTensor((self.tensor >> item) & 1)

    def get_tensor_bit(self, item):
        """
        Get a bit from the RingTensor.

        This method is similar to `get_bit`, except that the extracted result type is torch.

        Args:
            item (int): The index of the bit to be extracted.

        Returns:
            torch.Tensor: The extracted bit.

        Examples:
            >>> ring_tensor.get_tensor_bit(0)
        """
        assert (self.bit_len >> item >= 0), "bit index out of range"
        return (self.tensor >> item) & 1

    def reshape(self, *shape):
        """
        Reshape the current RingTensor instance.

        The method is used to reshape the current RingTensor, Create a new RingTensor instance new using the reshaped
        RingTensor and the current instance's dtype and device attributes.

        Args:
            *shape (torch.Size or tuple): A variable number of parameters, specifying a new shape.

        Returns:
            RingTensor: `self` after changing shape.

        Examples:
            >>> ring_tensor.reshape(2, 3)
        """
        new = self.__class__(self.tensor.reshape(*shape), dtype=self.dtype, device=self.device)
        new.bit_len = self.bit_len
        return new

    def img2col(self, k_size: int, stride: int):
        """
        Img2Col algorithm using PyTorch's unfold to speed up convolution and pooling.

        First, determine the shape of the img RingTensor, and retrieve the batch size, number of channels, height,
        and width. Then, The height and width of the output feature map are calculated based on the size of the input
        image, the size of the convolutional kernel, and the stride.

        Use PyTorch's unfold method to unfold the image on the specified dimensions (height and width) to get a
        RingTensor that contains all the convolution Windows, Then rearrange the dimensions and change the RingTensor shape.

        Args:
            k_size (int): The size of the kernel.
            stride (int): The stride of the convolution and pooling.

        Returns:
            tuple: The RingTensor object, the batch size of the image, the size of the col, The number of channels to input the image.

        Examples:
            >>> ring_tensor.img2col(3, 1)
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

        The method uses `torch.repeat_interleave` to repeat each element in the RingTensor according to the
        specified number of repeats along the given dimension.

        Args:
            repeats (int or list of int): The number of repetitions for each element.
            dim (int): The dimension along which to repeat the elements.

        Returns:
            RingTensor: A new RingTensor with the repeated elements.

        Examples:
            >>> ring_tensor.repeat_interleave(2, dim=0)
        """
        return self.__class__(self.tensor.repeat_interleave(repeats, dim), dtype=self.dtype, device=self.device)

    def repeat(self, *sizes):
        """
        Repeat the input RingTensor along each dimension specified by sizes.

        This method uses `torch.repeat` to replicate the RingTensor according to the provided sizes for each dimension.

        Args:
            *sizes (torch.Size or int): The number of times to repeat the RingTensor along each dimension.

        Returns:
            RingTensor: A new RingTensor with the repeated elements.

        Examples:
            >>> ring_tensor.repeat(2, 2)
        """
        return self.__class__(self.tensor.repeat(*sizes), dtype=self.dtype, device=self.device)

    def transpose(self, dim0, dim1):
        """
        Transpose the dimensions of the input RingTensor.

        The method uses `torch.transpose` to swap the specified dimensions (dim0 and dim1) of the RingTensor.

        Args:
            dim0 (int): The first dimension to swap.
            dim1 (int): The second dimension to swap.

        Returns:
            RingTensor: A new RingTensor with the transposed dimensions.

        Examples:
            >>> ring_tensor.transpose(0, 1)
        """
        return self.__class__(self.tensor.transpose(dim0, dim1), dtype=self.dtype, device=self.device)

    def pad(self, pad, mode='constant', value=0):
        """
        Pad the input RingTensor with a specified mode and value.

        The method uses `torch.nn.functional.pad` to add padding to the RingTensor. The type of padding is determined by the mode.

        Args:
            pad (tuple): The number of values to pad on each side of each dimension.
            mode (str): The padding mode (default is 'constant').
            value (int or float): The value to use for padding if mode is 'constant' (default is 0).

        Returns:
            RingTensor: A new RingTensor with the padded tensor.

        Examples:
            >>> ring_tensor.pad((1, 1))
        """
        return self.__class__(F.pad(self.tensor, pad, mode, value), dtype=self.dtype, device=self.device)

    def squeeze(self, dim=-1):
        """
        Remove dimensions of size 1 from the RingTensor.

        This method uses `torch.squeeze` to eliminate dimensions of size 1. If a specific dimension is provided,
        only that dimension is squeezed.

        Args:
            dim (int): The dimension to squeeze (default is -1, which means the last dimension).

        Returns:
            RingTensor: A new RingTensor with the squeezed dimensions.

        Examples:
            >>> ring_tensor.squeeze()
        """
        return self.__class__(self.tensor.squeeze(dim), dtype=self.dtype, device=self.device)

    def unsqueeze(self, dim=-1):
        """
        Add a dimension of size 1 to the RingTensor at the specified position.

        This method uses `torch.unsqueeze` to add a new dimension at the specified index.

        Args:
            dim (int): The index at which to insert the new dimension (default is -1, which means at the last position).

        Returns:
            RingTensor: A new RingTensor with the added dimension.

        Examples:
            >>> ring_tensor.unsqueeze(0)
        """
        return self.__class__(self.tensor.unsqueeze(dim), dtype=self.dtype, device=self.device)

    def size(self, dim=None):
        """
        Get the size of the RingTensor along the specified dimension.

        If dim is not provided, the entire shape of the RingTensor is returned.

        Args:
            dim (int or None): The dimension to get the size of (default is None).

        Returns:
            int or tuple: The size of the specified dimension or the full shape of the tensor.

        Examples:
            >>> ring_tensor.size()
        """
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    def view(self, *args):
        """
        Reshape the RingTensor to the specified dimensions.

        The method uses `torch.view` to create a new RingTensor with the desired shape. It maintains the data
        layout in memory.

        Args:
            *args (tuple[Any ...]): The desired shape for the output RingTensor.

        Returns:
            RingTensor: A new RingTensor with the specified shape.

        Examples:
            >>> ring_tensor.view(2, 3)
        """
        view = self.__class__(self.tensor.view(*args), dtype=self.dtype, device=self.device)
        view.bit_len = self.bit_len
        return view

    def flatten(self, start_dim=0, end_dim=-1):
        """
        Flatten the input RingTensor from start_dim to end_dim.

        This method uses `torch.flatten` to combine the specified dimensions into a single dimension.

        Args:
            start_dim (int): The first dimension to flatten (default is 0).
            end_dim (int): The last dimension to flatten (default is -1, which means the last dimension).

        Returns:
            RingTensor: A new RingTensor that has been flattened.

        Examples:
            >>> ring_tensor.flatten()
        """
        new = self.__class__(self.tensor.flatten(start_dim, end_dim), dtype=self.dtype, device=self.device)
        new.bit_len = self.bit_len
        return new

    def permute(self, *dims):
        """
        Permute the dimensions of the input RingTensor.

        This method uses `torch.permute` to rearrange the dimensions of the RingTensor according to the specified order.

        Args:
            *dims (tuple of int): The desired order of dimensions.

        Returns:
            RingTensor: A new RingTensor with the dimensions permuted.

        Examples:
            >>> ring_tensor.permute(1, 0)
        """
        new = self.__class__(self.tensor.permute(*dims), dtype=self.dtype, device=self.device)
        new.bit_len = self.bit_len
        return new

    def contiguous(self):
        """
        Return a contiguous RingTensor in memory.

        This method uses `torch.contiguous` to ensure that the RingTensor is stored in contiguous memory,
        which is often required for certain operations.

        Returns:
            RingTensor: A new RingTensor that is contiguous in memory.

        Examples:
            >>> ring_tensor.contiguous()
        """
        new = self.__class__(self.tensor.contiguous(), dtype=self.dtype, device=self.device)
        new.bit_len = self.bit_len
        return new

    def tolist(self):
        """
        Convert the RingTensor to a nested Python list.

        This method uses `torch.tolist` to transform the tensor into a standard nested list representation.

        Returns:
            list: A nested list representation of the RingTensor.

        Examples:
            >>> ring_tensor.tolist()
        """
        return self.tensor.tolist()

    def numel(self):
        """
        Get the total number of elements in the RingTensor.

        This method uses `torch.numel` to return the total count of elements in the RingTensor.

        Returns:
            int: The number of elements in the RingTensor.

        Examples:
            >>> ring_tensor.numel()
        """
        return self.tensor.numel()

    def signbit(self):
        """
        Get the sign bit of the RingTensor.

        This method uses `torch.signbit` to create a RingTensor indicating the sign of each element.
        It returns a tensor with bits set for negative values.

        Returns:
            RingTensor: A new RingTensor with the sign bits of the elements.

        Examples:
            >>> ring_tensor.signbit()
        """
        msb = torch.signbit(self.tensor) + 0
        return self.__class__(msb.to(data_type), self.dtype)

    def bit_slice(self, start, end):
        """
        Extract a slice of bits from the RingTensor.

        This method allows for bitwise slicing of the RingTensor, returning the bits from the specified start index to the end index.

        Note:
            The start index is inclusive, but the end index is exclusive.

        Args:
            start (int): The starting bit index.
            end (int): The ending bit index.

        Returns:
            RingTensor: A new RingTensor containing the specified slice of bits.

        Raises:
            AssertionError: If the bit indices are out of range.

        Examples:
            >>> ring_tensor.bit_slice(0, 8)
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

        This method uses `torch.expand` to create a new RingTensor with the specified sizes,
        broadcasting the original RingTensor's values as necessary.

        Args:
            *sizes (tuple of int): The desired sizes for each dimension.

        Returns:
            RingTensor: A new RingTensor with the expanded size.

        Examples:
            >>> ring_tensor.expand(2, 3)
        """
        return self.__class__(self.tensor.expand(*sizes), dtype=self.dtype, device=self.device)

    def argsort(self, dim=-1):
        """
        Sort the elements of the RingTensor along a specified dimension and return the indices of the sorted elements.

        Args:
            dim (int): The dimension along which to sort the elements (default is -1, which means the last dimension).

        Returns:
            RingTensor: The indices of the sorted elements.

        Examples:
            >>> ring_tensor.argsort()
        """
        return self.__class__(torch.argsort(self.tensor, dim=dim), dtype=self.dtype, device=self.device,
                              bit_len=self.bit_len)

    def index_select(self, dim, index):
        """
        Select elements from the RingTensor along a specified dimension using the provided indices.

        This method uses `torch.index_select` to gather elements from the RingTensor at the specified indices along the given dimension.

        Args:
            dim (int): The dimension along which to select elements.
            index (RingTensor or torch.Tensor): The indices of the elements to select.

        Returns:
            RingTensor: A new RingTensor containing the selected elements.

        Examples:
            >>> ring_tensor.index_select(0, index)
        """
        assert isinstance(index, (RingTensor, torch.Tensor)), "index must be RingTensor or torch.Tensor"
        if isinstance(index, RingTensor):
            index = index.tensor
        return self.__class__(torch.index_select(self.tensor, dim, index), dtype=self.dtype, device=self.device,
                              bit_len=self.bit_len)

    def index_add_(self, dim, index, source):
        """
        Add values from the source RingTensor to the specified indices in the current RingTensor.

        This method uses `torch.index_add_` to perform an in-place addition of values from the source RingTensor
        at the specified indices along the given dimension.

        Args:
            dim (int): The dimension along which to index.
            index (RingTensor or torch.Tensor): The indices at which to add values.
            source (RingTensor): The RingTensor containing values to add.

        Returns:
            RingTensor: The updated RingTensor after in-place addition.

        Examples:
            >>> ring_tensor.index_add_(0, index, source)
        """
        if isinstance(index, RingTensor):
            index = index.tensor
        self.tensor.index_add_(dim, index, source.tensor)
        return self

    def element_size(self):
        """
        Returns the size in bytes of an individual element.

        Returns:
            int: The size in bytes of an individual element.

        Examples:
            >>> ring_tensor.element_size()
        """
        return self.tensor.element_size()

    @staticmethod
    def where(condition, x, y):
        """
        Select elements from x or y based on a condition RingTensor.

        This method returns a RingTensor where elements are taken from x if the condition is true,
        and from y if false. Supports both RingTensor and int types for x and y.

        Args:
            condition (RingTensor): A RingTensor representing the condition.
            x (RingTensor or int): The RingTensor or scalar to select from when condition is true.
            y (RingTensor or int): The tensor or scalar to select from when condition is false.

        Returns:
            RingTensor: A new RingTensor with selected values based on the condition.

        Examples:
            >>> RingTensor.where(condition, x, y)
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

        Args:
            shape (tuple[int] or list[int]): The shape of the output RingTensor.
            dtype (str): The data type of the RingTensor (default is 'int').
            device (str): The device to allocate the RingTensor on (default is DEVICE).
            down_bound (int): The lower bound for random integers (inclusive).
            upper_bound (int): The upper bound for random integers (exclusive).

        Returns:
            RingTensor: A new RingTensor filled with random integers.

        Examples:
            >>> RingTensor.random((2, 3))
        """
        v = torch.randint(down_bound, upper_bound, shape, dtype=data_type, device=device)
        return RingTensor(v, dtype, device)

    @staticmethod
    def empty(size, dtype='int', device=DEVICE):
        """
        Create an uninitialized RingTensor of specified size.

        This method returns a RingTensor without initializing its values.

        Args:
            size (tuple[int] or list[int]): The size of the output RingTensor.
            dtype (str): The data type of the tensor (default is 'int').
            device (str): The device to allocate the tensor on (default is DEVICE).

        Returns:
            RingTensor: A new uninitialized RingTensor.

        Examples:
            >>> RingTensor.empty((2, 3))
        """
        return RingTensor(torch.empty(size, dtype=data_type), dtype, device)

    @staticmethod
    def empty_like(tensor):
        """
        Create an uninitialized RingTensor with the same shape as the input RingTensor.

        This method returns a RingTensor with the same shape as the provided RingTensor but without
        initializing its values.

        Note:
            The main difference between the `empty` and `empty_like` methods is:
                * `empty` creates a new RingTensor with a specified shape and data type.
                * `empty_like` creates a new RingTensor with the same shape as the given tensor.

        Args:
            tensor (RingTensor): The tensor to mimic.

        Returns:
            RingTensor: A new uninitialized RingTensor with the same shape.

        Raises:
            TypeError: If the input tensor is not a RingTensor.

        Examples:
            >>> RingTensor.empty_like(tensor)
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

        Args:
            size (tuple[int] or list[int]): The size of the output RingTensor.
            dtype (str): The data type of the tensor (default is 'int').
            device (str): The device to allocate the tensor on (default is DEVICE).

        Returns:
            RingTensor: A new RingTensor filled with zeros.

        Examples:
            >>> RingTensor.zeros((2, 3))
        """
        return RingTensor(torch.zeros(size, dtype=data_type, device=device), dtype)

    @staticmethod
    def zeros_like(tensor, dtype='int', device=DEVICE):
        """
        Create a RingTensor of zeros with the same shape as the input RingTensor.

        This method returns a RingTensor filled with zeros that has the same shape as the provided RingTensor.

        Note:
            The difference between `zeros` and `zeros_like` is as `empty` and `empty_like`.

        Args:
            tensor (RingTensor or torch.Tensor): The tensor to mimic.
            dtype (str): The data type of the tensor (default is 'int').
            device (str): The device to allocate the tensor on (default is DEVICE).

        Returns:
            RingTensor: A new RingTensor filled with zeros.

        Raises:
            TypeError: If the input tensor is neither a RingTensor nor a torch.Tensor.

        Examples:
            >>> RingTensor.zeros_like(tensor)
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

        Args:
            size (tuple of int): The size of the output RingTensor.
            dtype (str or dtype): The data type of the RingTensor (default is 'int').
            device (str): The device to allocate the RingTensor on (default is DEVICE).

        Returns:
            RingTensor: A new RingTensor filled with ones.

        Examples:
            >>> RingTensor.ones((2, 3))
        """
        scale = DTYPE_SCALE_MAPPING[dtype]
        return RingTensor(torch.ones(size, dtype=data_type, device=device) * scale, dtype)

    @staticmethod
    def ones_like(tensor, dtype='int', device=DEVICE):
        """
        Create a RingTensor of ones with the same shape as the input tensor.

        This method returns a RingTensor filled with ones that has the same shape as the provided tensor.

        Note:
            The difference between `ones` and `ones_like` is as `empty` and `empty_like`.

        Args:
            tensor (RingTensor or torch.Tensor): The tensor to mimic.
            dtype (str): The data type of the tensor (default is 'int').
            device (str): The device to allocate the tensor on (default is DEVICE).

        Returns:
            RingTensor: A new RingTensor filled with ones.

        Raises:
            TypeError: If the input tensor is neither a RingTensor nor a torch.Tensor.

        Examples:
            >>> RingTensor.ones_like(tensor)
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

        Args:
            size (tuple of int): The size of the output RingTensor.
            fill_value (Union): The value to fill the RingTensor with.
            device (str): The device to allocate the RingTensor on (default is DEVICE).

        Returns:
            RingTensor: A new RingTensor filled with the specified value.

        Examples:
            >>> RingTensor.full((2, 3), 5)
        """
        return RingTensor.convert_to_ring(torch.full(size, fill_value, device=device))

    @staticmethod
    def full_like(tensor, fill_value, device=DEVICE):
        """
        Create a RingTensor filled with a specified value with the same shape as the input tensor.

        This method initializes a RingTensor with the same shape as the provided tensor,
        filled with the given value.

        Args:
            tensor (RingTensor or torch.Tensor): The tensor to mimic.
            fill_value (Union): The value to fill the RingTensor with.
            device (str): The device to allocate the RingTensor on (default is DEVICE).

        Returns:
            RingTensor: A new RingTensor filled with the specified value.

        Raises:
            TypeError: If the input tensor is neither a RingTensor nor a torch.Tensor.

        Examples:
            >>> RingTensor.full_like(tensor, 5)
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
        spaced by the given `step`.

        Args:
            start (int): The starting value of the sequence.
            end (int): The end value of the sequence (exclusive).
            step (int): The spacing between consecutive values (default is 1).
            dtype (str): The data type of the output RingTensor.
            device (str): The device to allocate the RingTensor on (default is DEVICE).

        Returns:
            RingTensor: A new RingTensor containing the generated sequence.

        Examples:
            >>> RingTensor.arange(0, 10)
        """
        return RingTensor(torch.arange(start, end, step, dtype=data_type, device=device), dtype)

    @staticmethod
    def batch_randperm(num, n, device=DEVICE):
        """
        Generate a batch of random permutations of integers from 0 to n-1.

        Args:
            num (int): The number of permutations to generate.
            n (int): The upper bound (exclusive) for the range of integers.
            device (str): The device to allocate the RingTensor on (default is DEVICE).

        Returns:
            RingTensor: A RingTensor containing the batch of random permutations.

        Examples:
            >>> RingTensor.batch_randperm(5, 10)
        """
        rand = torch.rand(num, n, device=device)
        return RingTensor(rand.argsort(dim=1), dtype='int', device=device)
