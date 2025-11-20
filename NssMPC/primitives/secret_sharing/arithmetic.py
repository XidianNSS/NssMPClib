"""Arithmetic Secret Sharing
"""
import torch

from NssMPC.config import SCALE
# from NssMPC.protocols.semi_honest_2pc.base import add_public_value
# from NssMPC.protocols.honest_majority_3pc import v_mul, v_matmul, \
#     secure_ge as v_secure_ge, truncate
# from NssMPC.protocols.semi_honest_2pc import beaver_mul, secure_matmul, \
#     secure_div, secure_eq, secure_ge, secure_exp, secure_reciprocal_sqrt, truncate, secure_tanh
# from NssMPC.protocols.semi_honest_3pc import mul_with_out_trunc, matmul_with_out_trunc, \
#     secure_ge
from NssMPC.infra.mpc.party import PartyCtx, Party
from NssMPC.infra.tensor import RingTensor


# from NssMPC.runtime.party import HonestMajorityParty


#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

def auto_delegate(methods=None):
    def decorator(cls):
        def delegate_methods(method_name):
            if not hasattr(RingTensor, method_name):
                print(f"Warning: RingTensor does not have method '{method_name}'")
                return None

            def delegate(self, *args, **kwargs):
                result = getattr(self.item, method_name)(*args, **kwargs)
                if isinstance(result, (RingTensor, RingPair)):
                    return self.__class__(result)
                return result

            return delegate

        for name in methods or []:
            if (method := delegate_methods(name)) is not None:
                setattr(cls, name, method)
        return cls

    return decorator


def methods_delegate():
    def decorator(cls):

        def create_delegated_method(method_name):
            def delegate(self, *args, **kwargs):
                result_0 = getattr(self._item_0, method_name)(*args, **kwargs)
                result_1 = getattr(self._item_1, method_name)(*args, **kwargs)
                return cls(result_0, result_1)

            return delegate

        for name in RingTensor.__dict__.keys():
            if name not in cls.__dict__ and callable(getattr(RingTensor, name)):
                setattr(cls, name, create_delegated_method(name))

        return cls

    return decorator


@methods_delegate()
class RingPair:
    def __init__(self, item_0, item_1):
        """Initializes a RingPair instance with two RingTensor objects.

        Args:
            item_0 (RingTensor): The first RingTensor to be added to the RingPair instance.
            item_1 (RingTensor): The second RingTensor to be added.

        Examples:
            >>> rp = RingPair(rt1, rt2)
        """
        self._item_0 = item_0
        self._item_1 = item_1

    @property
    def dtype(self):
        """Gets the dtype of the RingTensor in this RingPair.

        Returns:
            type: The dtype of the RingTensors.

        Examples:
            >>> dt = rp.dtype
        """
        return self._item_0.dtype

    @dtype.setter
    def dtype(self, value):
        """Sets the dtype of the RingTensor in this object.

        Args:
            value (type): The new dtype to set.

        Examples:
            >>> rp.dtype = 'int64'
        """
        self._item_0.dtype = value
        self._item_1.dtype = value

    @property
    def device(self):
        """Gets the device on which the RingTensors in this RingPair are located.

        Returns:
            type: The device of the RingTensors.

        Examples:
            >>> dev = rp.device
        """
        return self._item_0.device

    @property
    def shape(self):
        """Gets the shape of the RingTensors in this RingPair.

        Returns:
            torch.Size: The shape of the first RingTensor.

        Examples:
            >>> s = rp.shape
        """
        return self._item_0.shape

    @property
    def scale(self):
        """Gets the scale of the RingTensors in this RingPair.

        Returns:
            int: The scale of the RingTensors.

        Examples:
            >>> sc = rp.scale
        """
        return self._item_0.scale

    @property
    def tensor(self):
        """Gets the stacked underlying tensors of this object.

        Returns:
            torch.Tensor: A stacked tensor containing both _item_0 and _item_1 tensors.

        Examples:
            >>> t = rp.tensor
        """
        return torch.stack((self._item_0.tensor, self._item_1.tensor))

    @property
    def T(self):
        """Gets the transposed RingPair of this object.

        Returns:
            RingPair: A new RingPair with transposed tensors.

        Examples:
            >>> rp_t = rp.T
        """
        return RingPair(self._item_0.T, self._item_1.T)

    def numel(self):
        """Gets the number of elements in the first RingTensor of this RingPair.

        Returns:
            int: The number of elements in the first RingTensor.

        Examples:
            >>> n = rp.numel()
        """
        return self._item_0.numel()

    def img2col(self, k_size: int, stride: int):
        """Performs img2col (image to column) operation on both tensors in the RingPair.

        Args:
            k_size: Kernel size for the img2col operation.
            stride: Stride size for the img2col operation.

        Returns:
            tuple: A tuple containing (RingPair, batch_size, out_size, channels).

        Examples:
            >>> res_rp, batch, out, ch = rp.img2col(3, 1)
        """
        res0, _, _, _ = self._item_0.img2col(k_size, stride)
        res1, batch, out_size, channel = self._item_1.img2col(k_size, stride)
        return RingPair(res0, res1), batch, out_size, channel

    def __str__(self):
        """Returns a string representation of the RingPair.

        Returns:
            str: A string showing the values of _item_0 and _item_1.

        Examples:
            >>> print(rp)
        """
        return f"[value0:{self._item_0}\nvalue1:{self._item_1}]"

    def __getitem__(self, item):
        """Gets the RingTensor in this RingPair by index.

        Args:
            item (int): The index (0 or 1) to retrieve the corresponding RingTensor.

        Returns:
            RingTensor: The RingTensor at the specified index.

        Raises:
            IndexError: If the index is not 0 or 1.

        Examples:
            >>> rt = rp[0]
        """
        assert item in [0, 1], IndexError("Index out of range")
        return self._item_0 if item == 0 else self._item_1

    def __setitem__(self, key, value):
        """Sets the RingTensor in this RingPair by index.

        Args:
            key (int): The index (0 or 1) to set the corresponding RingTensor.
            value (RingTensor): The new RingTensor to set at the specified index.

        Raises:
            IndexError: If the index is not 0 or 1.

        Examples:
            >>> rp[0] = new_rt
        """
        assert key in [0, 1], IndexError("Index out of range")
        if key == 0:
            self._item_0 = value
        else:
            self._item_1 = value


@auto_delegate(
    methods=['numel', 'tolist', '__neg__', '__xor__', 'reshape', 'view', 'transpose', 'squeeze', 'unsqueeze', 'flatten',
             'clone', 'pad', 'sum', 'repeat', 'repeat_interleave', 'permute', 'to', 'contiguous', 'expand',
             'index_select', 'load_from_file'])
class SecretSharingBase:
    """Base class for arithmetic operations with auto-delegation.

    This base class is designed for arithmetic operations, where most properties and methods
    are delegated to the `RingTensor` class. It serves as the foundation for ASS (Arithmetic Secret Sharing)
    and RSS (Replicated Secret Sharing) classes, providing basic arithmetic functionality.

    Args:
        item (RingTensor or RingPair): The `RingTensor` instance used for arithmetic operations.
    """

    def __init__(self, item):
        """Initialize an ArithmeticBase instance.

        Args:
            item (RingTensor or RingPair): The `RingTensor` instance used for arithmetic operations.

        Examples:
            >>> ss = SecretSharingBase(item)
        """
        self.item = item

    @property
    def dtype(self) -> str:
        """Gets the dtype of this object.

        Returns:
            type: The dtype.

        Examples:
            >>> dt = ss.dtype
        """
        return self.item.dtype

    @dtype.setter
    def dtype(self, value):
        """Sets the dtype of this object.

        Args:
            value (type): The new dtype.

        Examples:
            >>> ss.dtype = 'int64'
        """
        self.item.dtype = value

    @property
    def device(self) -> str:
        """Gets the device of this object.

        Returns:
            type: The device.

        Examples:
            >>> dev = ss.device
        """
        return self.item.device

    @property
    def shape(self):
        """Gets the shape of this object.

        Returns:
            torch.Size: The shape.

        Examples:
            >>> s = ss.shape
        """
        return self.item.shape

    @property
    def scale(self):
        """Gets the scale of this object.

        Returns:
            int: The scale.

        Examples:
            >>> sc = ss.scale
        """
        return self.item.scale

    @property
    def bit_len(self):
        """Gets the bit length of this object.

        Returns:
            int: The bit length.

        Examples:
            >>> bl = ss.bit_len
        """
        return self.item.bit_len

    @bit_len.setter
    def bit_len(self, value):
        """Sets the bit lengths of this object.

        Args:
            value (int): The new bit length.

        Examples:
            >>> ss.bit_len = 64
        """
        self.item.bit_len = value

    @property
    def T(self):
        """Returns an ASS instance with its dimensions reversed.

        Returns:
            SecretSharingBase: Transposed instance.

        Examples:
            >>> ss_t = ss.T
        """
        return self.__class__(self.item.T)

    def __getitem__(self, item):
        """Enables indexing for the ASS instance.

        Args:
            item (int): The position of the element to retrieve.

        Returns:
            SecretSharingBase: The element at the specified index.

        Examples:
            >>> elem = ss[0]
        """
        return self.__class__(self.item[item])

    def __setitem__(self, key, value):
        """Allows assignment to a specific index in the ASS instance.

        Args:
            key (int): The position of the element to be set.
            value (SecretSharingBase): The value to assign to the specified index.

        Raises:
            TypeError: If `value` is not of a supported type.

        Examples:
            >>> ss[0] = other_ss
        """
        if isinstance(value, self.__class__):
            self.item[key] = value.item.clone()
        else:
            raise TypeError(f"unsupported operand type(s) for setitem '{type(self)}' and {type(value)}")

    def __str__(self):
        """Returns a string representation of the ASS instance.

        Returns:
            str: A string that represents the ASS instance.

        Examples:
            >>> print(ss)
        """
        return f"{self.__class__.__name__}[\n{self.item}\n ]"  # party:{runtime.party.party_id if runtime}\n

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
    def load_from_file(cls, file_path):
        """Loads a ArithmeticBase object from a file.

        Args:
            file_path (str): The path from where the object should be loaded.

        Returns:
            RingTensor: The ArithmeticBase object loaded from the file.

        Examples:
            >>> obj = SecretSharingBase.load_from_file("path/to/file")
        """
        result = RingTensor.load_from_file(file_path)
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
        """Computes the maximum value in the input `x` along a specified dimension or over the entire input.

        This method applies an iterative approach to find the maximum by comparing pairs of elements.
        If **dim** is not specified, this method will return the maximum value over the entire input.

        Args:
            x (SecretSharingBase): Input on which to compute the maximum.
            dim (int, optional): The dimension along which to compute the maximum. If None, computes the maximum over the entire tensor. Defaults to None.

        Returns:
            SecretSharingBase: The maximum value(s) in the tensor along the specified dimension or overall.

        Examples:
            >>> m = SecretSharingBase.max(x, dim=0)
        """

        def max_iterative(inputs):
            """Iteratively computes the maximum by comparing pairs of elements.

            Args:
                inputs (SecretSharingBase): The input for which the maximum will be computed iteratively.

            Returns:
                SecretSharingBase: The maximum value from the input tensor.
            """
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

    def index_select(self, dim, index):
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
        """Returns the size of the object.

        If a specific dimension is provided, it returns the size along that dimension.
        Otherwise, it returns the shape of the entire object.

        Args:
            dim (int, optional): The dimension along which to get the size. If None, returns the full shape of the tensor. Defaults to None.

        Returns:
            tuple or int: The size of the object or the size along the specified dimension.

        Examples:
            >>> s = ss.size()
        """
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


class AdditiveSecretSharing(SecretSharingBase):
    """A class for arithmetic secret sharing over a RingTensor.

    This class extends ArithmeticBase and provides methods for performing
    secret sharing, as well as arithmetic operations (addition, subtraction,
    multiplication, etc.) **on secret-shared values**.

    Args:
        ring_tensor (RingTensor): The tensor used for secret sharing.
    """

    def __init__(self, ring_tensor):
        """Initializes an ASS object.

        Args:
            ring_tensor (RingTensor): The RingTensor used for secret sharing.

        Examples:
            >>> ass = AdditiveSecretSharing(rt)
        """
        assert isinstance(ring_tensor, RingTensor), "ring_tensor must be a RingTensor"
        super().__init__(ring_tensor)

    @property
    def ring_tensor(self):
        """Gets ring_tensor from an ASS instance.

        Returns:
            RingTensor: The underlying RingTensor.

        Examples:
            >>> rt = ass.ring_tensor
        """
        return self.item

    def __getitem__(self, item):
        """Enables indexing for the ASS instance.

        Args:
            item (int): The position of the element to retrieve.

        Returns:
            AdditiveSecretSharing: The element at the specified index from the original ASS instance.

        Examples:
            >>> val = ass[0]
        """
        return AdditiveSecretSharing(self.item[item])

    def __setitem__(self, key, value):
        """Allows assignment to a specific index in the ASS instance.

        Args:
            key (int): The position of the element to be set.
            value (AdditiveSecretSharing): The value to assign to the specified index.

        Raises:
            TypeError: If `value` is not of a supported type(`ASS`).

        Examples:
            >>> ass[0] = other_ass
        """
        if isinstance(value, AdditiveSecretSharing):
            self.item[key] = value.item.clone()
        else:
            raise TypeError(f"unsupported operand type(s) for setitem '{type(self)}' and {type(value)}")

    def __add__(self, other):
        """Adds an ASS object with another object.

        Args:
            other (AdditiveSecretSharing or RingTensor or int or float): The object to add.

        Returns:
            AdditiveSecretSharing: The result of addition.

        Raises:
            TypeError: If `other` is not of a supported type.

        Examples:
            >>> res = ass + other
        """
        if isinstance(other, AdditiveSecretSharing):
            return AdditiveSecretSharing(self.item + other.item)
        elif isinstance(other, RingTensor):  # for RingTensor or const number, only party 0 add it to the share tensor
            return self._add_public_value(other)
        elif isinstance(other, (int, float)):
            return self + RingTensor.convert_to_ring(int(other * self.scale))
        else:
            raise TypeError(f"unsupported operand type(s) for + '{type(self)}' and {type(other)}")

    __radd__ = __add__

    def __iadd__(self, other):
        """In-place addition.

        Args:
            other (AdditiveSecretSharing or RingTensor or int or float): The object to add.

        Returns:
            AdditiveSecretSharing: The result of in-place addition.

        Raises:
            TypeError: If `other` is not of a supported type.

        Examples:
            >>> ass += other
        """
        if isinstance(other, AdditiveSecretSharing):
            self.item += other.item
        elif isinstance(other, RingTensor):
            self.item = self._add_public_value(other).item
        elif isinstance(other, (int, float)):
            self.item += RingTensor.convert_to_ring(int(other * self.scale))
        else:
            raise TypeError(f"unsupported operand type(s) for += '{type(self)}' and {type(other)}")
        return self

    def __sub__(self, other):
        """Subtracts an object from an ASS object.

        Args:
            other (AdditiveSecretSharing or RingTensor or int or float): The object to subtract.

        Returns:
            AdditiveSecretSharing: The result of subtraction.

        Raises:
            TypeError: If `other` is not of a supported type.

        Examples:
            >>> res = ass - other
        """
        if isinstance(other, AdditiveSecretSharing):
            new_tensor = self.item - other.item
            return AdditiveSecretSharing(new_tensor)
        elif isinstance(other, RingTensor):
            return self._add_public_value(-other)
        elif isinstance(other, (int, float)):
            return self - RingTensor.convert_to_ring(int(other * self.scale))
        else:
            raise TypeError(f"unsupported operand type(s) for - '{type(self)}' and {type(other)}")

    def __rsub__(self, other):
        return -(self - other)

    def __isub__(self, other):
        """In-place subtraction.

        Args:
            other (AdditiveSecretSharing or RingTensor or int or float): The object to subtract.

        Returns:
            AdditiveSecretSharing: The result of in-place subtraction.

        Raises:
            TypeError: If `other` is not of a supported type.

        Examples:
            >>> ass -= other
        """
        if isinstance(other, AdditiveSecretSharing):
            self.item -= other.item
        elif isinstance(other, RingTensor):
            self.item = self._add_public_value(-other).item
        elif isinstance(other, (int, float)):
            self.item -= RingTensor.convert_to_ring(int(other * self.scale))
        else:
            raise TypeError(f"unsupported operand type(s) for -= '{type(self)}' and {type(other)}")
        return self

    def __mul__(self, other):
        """Multiplies an ASS object with a corresponding type.

        If other is an ASS instance, it performs a multiplication using beaver_mul.

        Args:
            other (AdditiveSecretSharing or RingTensor or int or float): The object to be multiplied.

        Returns:
            AdditiveSecretSharing: A new ASS instance representing the result.

        Raises:
            TypeError: If `other` is not of a supported type.

        Examples:
            >>> res = ass * other
        """
        if isinstance(other, AdditiveSecretSharing):
            res = AdditiveSecretSharing._mul(self, other)
        elif isinstance(other, RingTensor):
            res = AdditiveSecretSharing(RingTensor.mul(self.item, other))
        elif isinstance(other, int):
            return AdditiveSecretSharing(self.item * other)
        elif isinstance(other, float):
            assert self.dtype == 'float', "only float can calculate with float"
            res = AdditiveSecretSharing(self.item * int(other * self.scale))
        else:
            raise TypeError(f"unsupported operand type(s) for * '{type(self)}' and {type(other)}")

        return res / self.scale

    __rmul__ = __mul__

    def __matmul__(self, other):
        """Performs matrix multiplication between an ASS object with a corresponding type.

        If other is an ASS instance, it performs a matrix multiplication using
        the secure matrix multiplication protocol.

        Args:
            other (AdditiveSecretSharing or RingTensor): The object to perform matrix multiplication.

        Returns:
            AdditiveSecretSharing: A new ASS instance representing the result.

        Raises:
            TypeError: If `other` is not of a supported type.

        Examples:
            >>> res = ass @ other
        """
        if isinstance(other, AdditiveSecretSharing):
            res = self._matmul(other)
        elif isinstance(other, RingTensor):
            res = AdditiveSecretSharing(RingTensor.matmul(self.item, other))
        else:
            raise TypeError(f"unsupported operand type(s) for @ '{type(self)}' and {type(other)}")

        return res / self.scale

    def __rmatmul__(self, other):
        if isinstance(other, AdditiveSecretSharing):
            res = other._matmul(self)
        elif isinstance(other, RingTensor):
            res = AdditiveSecretSharing(RingTensor.matmul(other, self.item))
        else:
            raise TypeError(f"unsupported operand type(s) for @ '{type(self)}' and {type(other)}")

        return res / self.scale

    def __pow__(self, power, modulo=None):
        """Raises the ASS instance to the power of `power`.

        This method performs exponentiation on the current instance using the
        given `power`. Optionally, results can be reduced modulo `modulo` if provided.

        Args:
            power (int): The exponent to which the instance will be raised.
            modulo (int, optional): The modulo value to reduce the result. Defaults to None.

        Returns:
            AdditiveSecretSharing: A new ASS instance representing the result of exponentiation.

        Raises:
            TypeError: If `power` is not an integer.

        Examples:
            >>> res = ass ** 2
        """
        # TODO: 'continue' coming soon.
        if isinstance(power, int):
            temp = self
            res = 1
            while power:
                if power % 2 == 1:
                    res = temp * res
                temp = temp * temp
                power //= 2
            return res
        else:
            raise TypeError(f"unsupported operand type(s) for ** '{type(self)}' and {type(power)}")

    def __truediv__(self, other):
        """Divides the ASS instance by `other`.

        This method supports division by an integer, float, another ASS
        instance, or a RingTensor. If the other is an ASS instance, it performs a division using
        the secure division protocol.

        Args:
            other (int or float or AdditiveSecretSharing or RingTensor): The value by which to divide.

        Returns:
            AdditiveSecretSharing: A new ASS instance representing the result of division.

        Raises:
            TypeError: If `other` is not of a supported type.

        Examples:
            >>> res = ass / 2
        """
        if isinstance(other, int):
            if other == 1:
                return self
            return self._trunc(other)
        elif isinstance(other, float):
            if other == 1:
                return self
            from NssMPC.config import float_scale
            return (self / int(other * float_scale)) * float_scale
        elif isinstance(other, AdditiveSecretSharing):
            return self._div(other)
        elif isinstance(other, RingTensor):
            return (self * other.scale)._trunc(other.tensor)
        else:
            raise TypeError(f"unsupported operand type(s) for / '{type(self)}' and {type(other)}")

    def __eq__(self, other):
        """Compares the ASS instance for equality with `other`.

        This method compares the current instance with another ASS instance,
        a RingTensor, or an integer/float with the secure equality protocol.

        Args:
            other (AdditiveSecretSharing or RingTensor or int or float): The object to compare for equality.

        Returns:
            AdditiveSecretSharing: The corresponding element will be 1 if the two values are equal, otherwise 0.

        Raises:
            TypeError: If `other` is not of a supported type.

        Examples:
            >>> res = ass == other
        """
        if isinstance(other, (AdditiveSecretSharing, RingTensor)):
            return self._eq(other)
        elif isinstance(other, int):
            return self._eq(RingTensor.convert_to_ring(int(other * self.scale)))
        elif isinstance(other, float):
            assert self.dtype == 'float', "only float can compare with float"
            return self._eq(RingTensor.convert_to_ring(other))
        else:
            raise TypeError(f"unsupported operand type(s) for comparison '{type(self)}' and {type(other)}")

    def __ge__(self, other):
        """Compares if the ASS instance is greater than or equal to `other`.

        This method compares the current instance with another ASS instance,
        a RingTensor, or an integer/float for a greater than or equal relationship with the secure_ge protocol.

        Args:
            other (AdditiveSecretSharing or RingTensor or int or float): The object to compare against.

        Returns:
            AdditiveSecretSharing: The corresponding element will be 1 if the two values are equal, otherwise 0.

        Raises:
            TypeError: If `other` is not of a supported type.

        Examples:
            >>> res = ass >= other
        """
        if isinstance(other, (AdditiveSecretSharing, RingTensor)):
            return self._ge(other)
        elif isinstance(other, int):
            return self._ge(RingTensor.convert_to_ring(int(other * self.scale)))
        elif isinstance(other, float):
            assert self.dtype == 'float', "only float can compare with float"
            return self._ge(RingTensor.convert_to_ring(other))
        else:
            raise TypeError(f"unsupported operand type(s) for comparison '{type(self)}' and {type(other)}")

    def __le__(self, other):
        """Compares if the ASS instance is less than or equal to `other`.

        This method performs the comparison by negating the `__ge__` logic.

        Args:
            other (AdditiveSecretSharing or RingTensor or int or float): The object to compare against.

        Returns:
            AdditiveSecretSharing: The corresponding element will be 1 if the two values are equal, otherwise 0.

        Examples:
            >>> res = ass <= other
        """
        return -self >= -other

    def __gt__(self, other):
        """Compares if the ArithmeticSecretSharing instance is greater than `other`.

        This method is based on the `__le__` comparison.

        Args:
            other (AdditiveSecretSharing or RingTensor or int or float): The object to compare against.

        Returns:
            AdditiveSecretSharing: The corresponding element will be 1 if the two values are equal, otherwise 0.

        Examples:
            >>> res = ass > other
        """
        le = self <= other
        return -(le - 1)

    def __lt__(self, other):
        """Compares if the ArithmeticSecretSharing instance is less than `other`.

        This method is based on the `__ge__` comparison.

        Args:
            other (AdditiveSecretSharing or RingTensor or int or float): The object to compare against.

        Returns:
            AdditiveSecretSharing: The corresponding element will be 1 if the two values are equal, otherwise 0.

        Examples:
            >>> res = ass < other
        """
        ge = self >= other
        return -(ge - 1)

    @classmethod
    def cat(cls, tensor_list, dim=0):
        """Concatenates a list of ASS instances along a given dimension.

        This method concatenates the `RingTensor` objects inside each ASS
        instance in the list along the specified dimension.

        Args:
            tensor_list (list[AdditiveSecretSharing]): A list of ASS instances to concatenate.
            dim (int, optional): The dimension along which to concatenate the tensors. Defaults to 0.

        Returns:
            AdditiveSecretSharing: A new ASS instance with concatenated RingTensors.

        Examples:
            >>> res = AdditiveSecretSharing.cat([ass1, ass2], dim=0)
        """
        result = RingTensor.cat([e.item for e in tensor_list], dim)
        return cls(result)

    @classmethod
    def stack(cls, tensor_list, dim=0):
        """Stacks a list of ASS instances along a new dimension.

        This method stacks the `RingTensor` objects inside each ASS
        instance in the list along a new dimension specified by `dim`.

        Args:
            tensor_list (list[AdditiveSecretSharing]): A list of ASS instances to stack.
            dim (int, optional): The dimension along which to stack the tensors. Defaults to 0.

        Returns:
            AdditiveSecretSharing: A new ASS instance with stacked tensors.

        Examples:
            >>> res = AdditiveSecretSharing.stack([ass1, ass2], dim=0)
        """
        result = RingTensor.stack([e.item for e in tensor_list], dim)
        return cls(result)

    @classmethod
    def gather(cls, input, dim, index):
        """Gathers elements from the ASS instance along an axis specified by `dim`.

        This method gathers elements from `input` along the specified dimension `dim`,
        using the indices provided by `index`.

        Args:
            input (AdditiveSecretSharing): The ASS instance from which to gather elements.
            dim (int): The dimension along which to gather.
            index (Tensor): The indices of the elements to gather.

        Returns:
            AdditiveSecretSharing: A new ASS instance with gathered elements.

        Examples:
            >>> res = AdditiveSecretSharing.gather(ass, 0, idx)
        """
        result = RingTensor.gather(input.item, dim, index)
        return cls(result)

    @classmethod
    def roll(cls, input, shifts, dims=0):
        """Rolls the elements of an ASS instance along a specified dimension.

        This method rolls the elements of the `RingTensor` in the specified `input` along
        the `dims` dimension by the number of positions specified in `shifts`.

        Args:
            input (AdditiveSecretSharing): The ASS instance whose elements will be rolled.
            shifts (int): The number of positions by which the elements are shifted.
            dims (int, optional): The dimension along which to roll the elements. Defaults to 0.

        Returns:
            AdditiveSecretSharing: A new ASS instance with the rolled elements.

        Examples:
            >>> res = AdditiveSecretSharing.roll(ass, 1, 0)
        """
        result = RingTensor.roll(input.item, shifts, dims)
        return cls(result)

    @classmethod
    def rotate(cls, input, shifts):
        """Rotates each row of a 2-dimensional ASS instance to the left or right.

        This method rotates each row of the 2D `RingTensor` in the specified `input` by the
        number of positions specified in `shifts`.

        Args:
            input (AdditiveSecretSharing): The 2-dimensional ASS instance to rotate.
            shifts (int): The number of positions to rotate each row. Positive values rotate to the right, negative to the left.

        Returns:
            AdditiveSecretSharing: A new ASS instance with the rotated rows.

        Examples:
            >>> res = AdditiveSecretSharing.rotate(ass, 1)
        """
        result = RingTensor.rotate(input.item, shifts)
        return cls(result)

    @staticmethod
    def restore_from_shares(share_0, share_1):
        """Recovers the original data from two ASS instance(or we could say two shares).

        Args:
            share_0 (AdditiveSecretSharing): The first share.
            share_1 (AdditiveSecretSharing): The second share.

        Returns:
            RingTensor: The original data recovered from the two shares.

        Examples:
            >>> res = AdditiveSecretSharing.restore_from_shares(s0, s1)
        """
        return share_0.item + share_1.item

    def restore(self, party: Party = None):
        """Restores the original data from secret shares.

        This method is used to reconstruct the original data by combining the shares
        held by different parties. It requires communication between the parties.

        Args:
            party (Party, optional): The party context. Defaults to None.

        Returns:
            RingTensor: The reconstructed original data as a RingTensor.

        Examples:
            >>> res = ass.restore()
        """
        if party is None:
            party = PartyCtx.get()
        party.send(self)
        other = party.recv()
        return self.item + other.item

    @classmethod
    def share(cls, tensor: RingTensor, num_of_party: int = 2):
        """Performs secret sharing on a RingTensor across multiple parties.

        This method divides the given `RingTensor` into secret shares, distributing
        the shares among the specified number of parties. The returned shares
        are still RingTensors, not ASS objects.

        Args:
            tensor (RingTensor): The RingTensor to be secret-shared.
            num_of_party (int, optional): The number of parties among which the tensor will be shared. Defaults to 2.

        Returns:
            list[AdditiveSecretSharing]: A list of ArithmeticSecretSharing shares, one for each party.

        Examples:
            >>> shares = AdditiveSecretSharing.share(rt, 2)
        """
        share = []
        x_0 = tensor.clone()

        for i in range(num_of_party - 1):
            x_i = RingTensor.random(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            share.append(cls(x_i))
            x_0 -= x_i
        share.append(cls(x_0))
        return share


class ReplicatedSecretSharing(SecretSharingBase):
    """A class for replicated secret sharing over a RingPair.

    This class extends ArithmeticBase and provides methods for performing
    3PC(Secure 3-Party Computation) replicated secret sharing, as well as arithmetic operations (addition, subtraction,
    multiplication, etc.) **on 3PC secret-shared values**.

    Args:
        ring_pair (RingPair or list[RingTensor]): The RingTensor pair(RingPair) used for secret sharing.
    """

    def __init__(self, ring_pair):
        """Initializes an ReplicatedSecretSharing object.

        Args:
            ring_pair (RingPair or list[RingTensor]): The RingTensor pair used for secret sharing.

        Examples:
            >>> rss = ReplicatedSecretSharing(rp)
        """
        if isinstance(ring_pair, list):
            ring_pair = RingPair(ring_pair[0], ring_pair[1])
        super(ReplicatedSecretSharing, self).__init__(ring_pair)

    def __getitem__(self, item):
        """Enables indexing for the RSS instance.

        Args:
            item (int): The position of the element to retrieve.

        Returns:
            ReplicatedSecretSharing: The element at the specified index from the original RSS instance.

        Examples:
            >>> val = rss[0]
        """
        return ReplicatedSecretSharing([self.item[0][item], self.item[1][item]])

    def __setitem__(self, key, value):
        """Allows assignment to a specific index in the RSS instance.

        Args:
            key (int): The position of the element to be set.
            value (ReplicatedSecretSharing): The value to assign to the specified index.

        Raises:
            TypeError: If `value` is not of a supported type(`RSS`).

        Examples:
            >>> rss[0] = other_rss
        """
        if isinstance(value, ReplicatedSecretSharing):
            self.item[0][key] = value.item[0].clone()
            self.item[1][key] = value.item[1].clone()
        else:
            raise TypeError(f"unsupported operand type(s) for setitem '{type(self)}' and {type(value)}")

    def __str__(self):
        """Returns a string representation of the RSS instance.

        Returns:
            str: A string that represents the RSS instance.

        Examples:
            >>> print(rss)
        """
        return f"[{self.__class__.__name__}\n {self.item}]"

    def __add__(self, other):
        """Adds an RSS object with a corresponding type.

        Args:
            other (ReplicatedSecretSharing or RingTensor or int or float): The object to be added.

        Returns:
            ReplicatedSecretSharing: A new RSS instance representing the result.

        Raises:
            TypeError: If `other` is not of a supported type.

        Examples:
            >>> res = rss + other
        """
        if isinstance(other, ReplicatedSecretSharing):
            return ReplicatedSecretSharing(RingPair(self.item[0] + other.item[0], self.item[1] + other.item[1]))
        elif isinstance(other, RingTensor):
            return self._add_public_value(other)
        elif isinstance(other, (int, float)):
            return self + RingTensor.convert_to_ring(int(other * self.scale))
        else:
            raise TypeError("unsupported operand type(s) for + 'ReplicatedSecretSharing' and ", type(other))

    __radd__ = __add__

    def __sub__(self, other):
        """Subtracts an RSS object with a corresponding type.

        Args:
            other (ReplicatedSecretSharing or RingTensor or int): The object to subtract.

        Returns:
            ReplicatedSecretSharing: A new RSS instance representing the result.

        Raises:
            TypeError: If `other` is not of a supported type.

        Examples:
            >>> res = rss - other
        """
        if isinstance(other, ReplicatedSecretSharing):
            return ReplicatedSecretSharing(RingPair(self.item[0] - other.item[0], self.item[1] - other.item[1]))
        elif isinstance(other, RingTensor):
            return self._add_public_value(-other)
        elif isinstance(other, int):
            return self - RingTensor.convert_to_ring(int(other * self.scale))
        else:
            raise TypeError("unsupported operand type(s) for - 'ReplicatedSecretSharing' and ", type(other))

    def __rsub__(self, other):
        return -(self - other)

    def __mul__(self, other):
        """Multiplies an RSS object with a corresponding type.

        If other is an RSS instance, it performs a multiplication using the v_mul method
        for security of the computation.

        Args:
            other (ArithmeticSecretSharing or RingTensor or int): The object to be multiplied.

        Returns:
            AdditiveSecretSharing: A new ASS instance representing the result.

        Raises:
            TypeError: If `other` is not of a supported type.

        Examples:
            >>> res = rss * other
        """
        if isinstance(other, ReplicatedSecretSharing):
            return self._mul(other)
        elif isinstance(other, RingTensor):
            result0 = RingTensor.mul(self.item[0], other)
            result1 = RingTensor.mul(self.item[1], other)
            result = ReplicatedSecretSharing(RingPair(result0, result1))
            if self.dtype == "float":
                return result._trunc()
            return result
        elif isinstance(other, int):
            result = self.item * other
            return ReplicatedSecretSharing(result)
        else:
            raise TypeError("unsupported operand type(s) for * 'ReplicatedSecretSharing' and ", type(other))

    __rmul__ = __mul__

    def __matmul__(self, other):
        """Performs matrix multiplication between an RSS object with a corresponding type.

        If other is an RSS instance, it performs a matrix multiplication using
        the secure matrix multiplication protocol for security of the computation.

        Args:
            other (ReplicatedSecretSharing or RingTensor): The object to perform matrix multiplication.

        Returns:
            ReplicatedSecretSharing: A new RSS instance representing the result.

        Raises:
            TypeError: If `other` is not of a supported type.

        Examples:
            >>> res = rss @ other
        """
        if isinstance(other, ReplicatedSecretSharing):
            return self._matmul(other)
        elif isinstance(other, RingTensor):
            result0 = RingTensor.matmul(self.item[0], other)
            result1 = RingTensor.matmul(self.item[1], other)
            result = ReplicatedSecretSharing(RingPair(result0, result1))
            torch.cuda.empty_cache()
            if self.dtype == "float":
                return result._trunc()
            return result
        else:
            raise TypeError("unsupported operand type(s) for @ 'ReplicatedSecretSharing' and ", type(other))

    def __rmatmul__(self, other):
        if isinstance(other, ReplicatedSecretSharing):
            return other.__matmul__(self)
        elif isinstance(other, RingTensor):
            # 手动处理 RingTensor @ RSS 的情况
            result0 = RingTensor.matmul(other, self.item[0])
            result1 = RingTensor.matmul(other, self.item[1])
            result = ReplicatedSecretSharing(RingPair(result0, result1))
            torch.cuda.empty_cache()
            if self.dtype == "float":
                return result._trunc()
            return result
        else:
            raise TypeError("unsupported operand type(s) for @ 'ReplicatedSecretSharing' and ", type(other))

    def __ge__(self, other):
        """Compares if the RSS instance is greater than or equal to `other`.

        This method compares the current instance with another RSS instance,
        a RingTensor, or an integer/float for a greater than or equal relationship with the secure_ge protocol.

        Args:
            other (ReplicatedSecretSharing or RingTensor or int or float): The object to compare against.

        Returns:
            ReplicatedSecretSharing: The corresponding element will be 1 if the two values are equal, otherwise 0.

        Raises:
            TypeError: If `other` is not of a supported type.

        Examples:
            >>> res = rss >= other
        """
        if isinstance(other, (ReplicatedSecretSharing, RingTensor)):
            return self._ge(other)
        elif isinstance(other, int):
            return self >= RingTensor.convert_to_ring(int(other * self.scale))
        elif isinstance(other, float):
            assert self.dtype == 'float', "only float can compare with float"
            return self >= RingTensor.convert_to_ring(other)
        else:
            raise TypeError(f"unsupported operand type(s) for comparison '{type(self)}' and {type(other)}")

    def __le__(self, other):
        """Compares if the RSS instance is less than or equal to `other`.

        This method performs the comparison by negating the `__ge__` logic.

        Args:
            other (ReplicatedSecretSharing or RingTensor or int or float): The object to compare against.

        Returns:
            ReplicatedSecretSharing: The corresponding element will be 1 if the two values are equal, otherwise 0.

        Examples:
            >>> res = rss <= other
        """
        return -self >= -other

    def __gt__(self, other):
        """Compares if the RSS instance is greater than `other`.

        This method is based on the `__le__` comparison.

        Args:
            other (ReplicatedSecretSharing or RingTensor or int or float): The object to compare against.

        Returns:
            ReplicatedSecretSharing: The corresponding element will be 1 if the two values are equal, otherwise 0.

        Examples:
            >>> res = rss > other
        """
        le = self <= other
        return -(le - 1)

    def __lt__(self, other):
        """Compares if the ArithmeticSecretSharing instance is less than `other`.

        This method is based on the `__ge__` comparison.

        Args:
            other (ReplicatedSecretSharing or RingTensor or int or float): The object to compare against.

        Returns:
            ReplicatedSecretSharing: The corresponding element will be 1 if the two values are equal, otherwise 0.

        Examples:
            >>> res = rss < other
        """
        ge = self >= other
        return -(ge - 1)

    @classmethod
    def cat(cls, tensor_list, dim=0):
        """Concatenates a list of RSS instances along a given dimension.

        This method concatenates the `RingPair` objects inside each RSS
        instance in the list along the specified dimension.

        Args:
            tensor_list (list[ReplicatedSecretSharing]): A list of RSS instances to concatenate.
            dim (int, optional): The dimension along which to concatenate the tensors. Defaults to 0.

        Returns:
            ReplicatedSecretSharing: A new RSS instance with concatenated RingTensors.

        Examples:
            >>> res = ReplicatedSecretSharing.cat([rss1, rss2], dim=0)
        """
        result_0 = RingTensor.cat([e.item[0] for e in tensor_list], dim)
        result_1 = RingTensor.cat([e.item[1] for e in tensor_list], dim)
        return cls(RingPair(result_0, result_1))

    @classmethod
    def stack(cls, tensor_list, dim=0):
        """Stacks a list of RSS instances along a new dimension.

        This method stacks the `RingPair` objects inside each RSS
        instance in the list along a new dimension specified by `dim`.

        Args:
            tensor_list (list[ReplicatedSecretSharing]): A list of RSS instances to stack.
            dim (int, optional): The dimension along which to stack the tensors. Defaults to 0.

        Returns:
            ReplicatedSecretSharing: A new RSS instance with stacked tensors.

        Examples:
            >>> res = ReplicatedSecretSharing.stack([rss1, rss2], dim=0)
        """
        result_0 = RingTensor.stack([e.item[0] for e in tensor_list], dim)
        result_1 = RingTensor.stack([e.item[1] for e in tensor_list], dim)
        return cls(RingPair(result_0, result_1))

    @classmethod
    def roll(cls, input, shifts, dims=0):
        """Rolls the elements of an RSS instance along a specified dimension.

        This method rolls the elements of the `RingPair` in the specified `input` along
        the `dims` dimension by the number of positions specified in `shifts`.

        Args:
            input (ReplicatedSecretSharing): The RSS instance whose elements will be rolled.
            shifts (int): The number of positions by which the elements are shifted.
            dims (int, optional): The dimension along which to roll the elements. Defaults to 0.

        Returns:
            ReplicatedSecretSharing: A new RSS instance with the rolled elements.

        Examples:
            >>> res = ReplicatedSecretSharing.roll(rss, 1, 0)
        """
        result_0 = RingTensor.roll(input.item[0], shifts, dims)
        result_1 = RingTensor.roll(input.item[1], shifts, dims)
        return cls(RingPair(result_0, result_1))

    @classmethod
    def rotate(cls, input, shifts):
        """Rotates each row of a 2-dimensional RSS instance to the left or right.

        This method rotates each row of the 2D `RingPair` in the specified `input` by the
        number of positions specified in `shifts`.

        Args:
            input (ReplicatedSecretSharing): The 2-dimensional ASS instance to rotate.
            shifts (int): The number of positions to rotate each row. Positive values rotate to the right, negative to the left.

        Returns:
            ReplicatedSecretSharing: A new RSS instance with the rotated rows.

        Examples:
            >>> res = ReplicatedSecretSharing.rotate(rss, 1)
        """
        result_0 = RingTensor.rotate(input.item[0], shifts)
        result_1 = RingTensor.rotate(input.item[1], shifts)
        return cls(RingPair(result_0, result_1))

    @staticmethod
    def gen_and_share(r_tensor, party):
        """Generates and shares the input r_tensor.

        Share the r_tenor to another two parties, and return one share of the r_tensor which
        should be kept by the current party.

        Args:
            r_tensor (RingTensor): The RingTensor to be shared.
            party (Party): The party that holds the RingPair.

        Returns:
            ReplicatedSecretSharing: The share should be kept by the current party.

        Examples:
            >>> share = ReplicatedSecretSharing.gen_and_share(rt, party)
        """
        r0, r1, r2 = ReplicatedSecretSharing.share(r_tensor)
        party.send((party.party_id + 1) % 3, r1)
        party.send((party.party_id + 2) % 3, r2)
        return r0

    @classmethod
    def share(cls, tensor: RingTensor):
        """Performs a three-party replicated secret sharing on the input tensor.

        Args:
            tensor (RingTensor): The RingTensor to be shared.

        Returns:
            list[ReplicatedSecretSharing]: The list of three RSS shares, and the attribute——Party of each share is None.

        Examples:
            >>> shares = ReplicatedSecretSharing.share(rt)
        """
        shares = []
        x_0 = RingTensor.random(tensor.shape, tensor.dtype, tensor.device)
        x_1 = RingTensor.random(tensor.shape, tensor.dtype, tensor.device)
        x_2 = tensor - x_0 - x_1
        shares.append(ReplicatedSecretSharing([x_0, x_1]))
        shares.append(ReplicatedSecretSharing([x_1, x_2]))
        shares.append(ReplicatedSecretSharing([x_2, x_0]))
        return shares

    def restore(self, party: Party = None):
        """Restores the original data from the secret shares.

        Perform the send-and-receive process to restore the original RingTensor.

        Returns:
            RingTensor: The restored RingTensor.

        Examples:
            >>> res = rss.restore()
        """
        # 发送部分
        if party is None:
            party = PartyCtx.get()
        party.send((party.party_id + 1) % 3, self.item[0])
        # 接收部分
        other = party.recv((party.party_id + 2) % 3)
        return self.item[0] + self.item[1] + other

    @staticmethod
    def reshare(value: RingTensor, party):
        """Reshares the shared values held by each party without altering the restored value.

        We use this method to convert the state where each party holds a single share
        into a state where they hold RSS shares, i.e., RingPair.

        Args:
            value (RingTensor): The share that the current party holds.
            party (Party): The party that holds the share.

        Returns:
            ReplicatedSecretSharing: The RSS instance that belongs to the current party.

        Examples:
            >>> rss = ReplicatedSecretSharing.reshare(val, party)
        """
        r_0 = party.prg_0.random(value.numel())
        r_1 = party.prg_1.random(value.numel())
        r_0 = r_0.reshape(value.shape)
        r_1 = r_1.reshape(value.shape)
        r_0.dtype = r_1.dtype = value.dtype
        value = value.tensor + r_0 - r_1
        party.send((party.party_id + 2) % 3, value)
        other = party.recv((party.party_id + 1) % 3)
        return ReplicatedSecretSharing(RingPair(value, other))

    @classmethod
    def random(cls, shape, party):
        """Generates RSS with a specified shape and random content.

        We use this method to generate a random RSS instance with a specified shape and party.

        Args:
            shape (tuple or list): The tuple representing the desired shape of the RSS.
            party (Party): The party that holds the random RSS.

        Returns:
            ReplicatedSecretSharing: The RSS instance with the specified shape and random content.

        Examples:
            >>> rss = ReplicatedSecretSharing.random((2, 3), party)
        """
        num_of_value = 1
        for d in shape:
            num_of_value *= d
        r_0 = party.prg_0.random(num_of_value)
        r_1 = party.prg_1.random(num_of_value)
        r = ReplicatedSecretSharing([r_0, r_1])
        r = r.reshape(shape)
        return r

    @classmethod
    def rand_like(cls, x, party):
        """Generates a random RSS with the same shape as input *x*.

        We use this method to generate a random RSS with the same shape as input *x*.

        Args:
            x (ReplicatedSecretSharing or RingTensor or torch.Tensor): The input whose shape will be used to generate the random RSS.
            party (Party): The party that holds the random RSS.

        Returns:
            ReplicatedSecretSharing: The RSS instance with the same shape as *x*.

        Examples:
            >>> rss = ReplicatedSecretSharing.rand_like(other, party)
        """
        r = ReplicatedSecretSharing.random(x.shape, party)
        if isinstance(x, (RingTensor, ReplicatedSecretSharing)):
            r.dtype = x.dtype
        return r

    def _add_public_value(self, other):
        raise NotImplementedError

    def _mul(self, other):
        raise NotImplementedError

    def _matmul(self, other):
        raise NotImplementedError

    def _div(self, other):
        raise NotImplementedError

    def _trunc(self, scale: int = SCALE):
        raise NotImplementedError

    def _eq(self, other):
        raise NotImplementedError

    def _ge(self, other):
        raise NotImplementedError

    def exp(self):
        raise NotImplementedError

    def rsqrt(self):
        raise NotImplementedError

    def tanh(self):
        raise NotImplementedError
