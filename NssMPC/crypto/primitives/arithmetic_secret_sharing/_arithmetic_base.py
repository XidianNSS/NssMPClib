#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch
from NssMPC.common.ring import RingTensor


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


@auto_delegate(
    methods=['numel', 'tolist', '__neg__', '__xor__', 'reshape', 'view', 'transpose', 'squeeze', 'unsqueeze', 'flatten',
             'clone',
             'pad', 'sum', 'repeat', 'repeat_interleave', 'permute', 'to', 'contiguous', 'expand', 'load_from_file'])
class SecretSharingBase:
    """
    Base class for arithmetic operations with auto-delegation.

    This base class is designed for arithmetic operations, where most properties and methods
    are delegated to the `RingTensor` class. It serves as the foundation for ASS (Arithmetic Secret Sharing)
    and RSS (Replicated Secret Sharing) classes, providing basic arithmetic functionality.

    :param item: The `RingTensor` instance used for arithmetic operations.
    :type item: RingTensor or RingPair
    :param party: The party or participant that owns the `ArithmeticBase` object.
    :type party: Party
    """

    def __init__(self, item):
        """
        Initialize an ArithmeticBase instance.

        We use this method to initialize a new ArithmeticBase instance.

        :param item: The `RingTensor` instance used for arithmetic operations.
        :type item: RingTensor or RingPair
        """
        self.item = item

    @property
    def dtype(self):
        """Get the dtype of this object."""
        return self.item.dtype

    @dtype.setter
    def dtype(self, value):
        """Set the dtype of this object."""
        self.item.dtype = value

    @property
    def device(self):
        """Set the device of this object."""
        return self.item.device

    @property
    def shape(self):
        """Get the shape of this object."""
        return self.item.shape

    @property
    def scale(self):
        """Get the scale of this object."""
        return self.item.scale

    @property
    def bit_len(self):
        """Get the bit length of this object."""
        return self.item.bit_len

    @bit_len.setter
    def bit_len(self, value):
        """Set the bit lengths of this object."""
        self.item.bit_len = value

    @property
    def T(self):
        """Returns an ASS instance with its dimensions reversed."""
        return self.__class__(self.item.T)

    def __getitem__(self, item):
        """
        Enables indexing for the ASS instance, allowing elements to be accessed
        using square brackets [ ] like a list.

        :param item: The position of the element to retrieve.
        :type item: int
        :returns: The element at the specified index from the original ASS instance
        :rtype: ArithmeticSecretSharing

        """
        return self.__class__(self.item[item])

    def __setitem__(self, key, value):
        """
        Allows assignment to a specific index in the ASS instance
        using square brackets [ ].

        :param key: The position of the element to be set.
        :type key: int
        :param value: The value to assign to the specified index.
        :type value: ArithmeticSecretSharing
        :returns: None
        :raises TypeError: If `value` is not of a supported type(`ASS`).
        """
        if isinstance(value, self.__class__):
            self.item[key] = value.item.clone()
        else:
            raise TypeError(f"unsupported operand type(s) for setitem '{type(self)}' and {type(value)}")

    def __str__(self):
        """
        Returns a string representation of the ASS instance.

        :returns: A string that represents the ASS instance.
        :rtype: str

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
        """
        TODO: 返回的是RingTensor，且未实现，是否正确。
        Load a ArithmeticBase object from a file.

        We use this method to load a ArithmeticBase object from a file, using the provided file path.

        :param file_path: The path from where the object should be loaded.
        :type file_path: str
        :returns: The ArithmeticBase object loaded from the file.
        :rtype: RingTensor

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
        """
        Computes the maximum value in the input `x` along a specified dimension or over the entire input.

        This method applies an iterative approach to find the maximum by comparing pairs of elements.
        If **dim** is not specified, this method will return the maximum value over the entire input.

        :param x: Input on which to compute the maximum.
        :type x: ASS or RSS
        :param dim: The dimension along which to compute the maximum. If None, computes the maximum over the entire tensor.
        :type dim: int, default to be *None*
        :return: The maximum value(s) in the tensor along the specified dimension or overall.
        :rtype: ASS or RSS

        .. hint::
            We implement this method in the base class for the ASS class and the RSS class.

        """

        def max_iterative(inputs):
            """
            Iteratively computes the maximum by comparing pairs of elements.

            :param inputs: The input for which the maximum will be computed iteratively.
            :type inputs: ASS or RSS
            :return: The maximum value from the input tensor.
            :rtype: ASS or RSS
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
        """
        Returns the size of the object. If a specific dimension is provided, it returns the size along that dimension.
        Otherwise, it returns the shape of the entire object.

        :param dim: The dimension along which to get the size. If None, returns the full shape of the tensor.
        :type dim: int, optional
        :return: The size of the object or the size along the specified dimension.
        :rtype: tuple or int

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
        """
        Initialize a RingPair instance with two RingTensor objects.

        :param item_0: The fist RingTensor to be added to the RingPair instance.
        :type item_0: RingTensor
        :param item_1: The second RingTensor to be added.
        :type item_1: RingTensor
        """
        self._item_0 = item_0
        self._item_1 = item_1

    @property
    def dtype(self):
        """
        Get the dtype of the RingTensor in this RingPair.

        :return: The dtype of the RingTensors.
        :rtype: Type
        """
        return self._item_0.dtype

    @dtype.setter
    def dtype(self, value):
        """
        Set the dtype of the RingTensor in this object.

        :param value: The new dtype to set.
        :type value: Type
        """
        self._item_0.dtype = value
        self._item_1.dtype = value

    @property
    def device(self):
        """
        Get the device on which the RingTensors in this RingPair are located.

        :return: The device of the RingTensors.
        :rtype: Type
        """
        return self._item_0.device

    @property
    def shape(self):
        """
        Get the shape of the RingTensors in this RingPair.

        :return: The shape of the first RingTensor.
        :rtype: torch.Size
        """
        return self._item_0.shape

    @property
    def scale(self):
        """
        Get the scale of the RingTensors in this RingPair.

        :return: The scale of the RingTensors.
        :rtype: int
        """
        return self._item_0.scale

    @property
    def tensor(self):
        """
        Get the stacked underlying tensors of this object.

        :return: A stacked tensor containing both _item_0 and _item_1 tensors.
        :rtype: torch.Tensor
        """
        return torch.stack((self._item_0.tensor, self._item_1.tensor))

    @property
    def T(self):
        """
        Get the transposed RingPair of this object.

        :return: A new RingPair with transposed tensors.
        :rtype: RingPair
        """
        return RingPair(self._item_0.T, self._item_1.T)

    def numel(self):
        """
        Get the number of elements in the first RingTensor of this RingPair.

        :return: The number of elements in the first RingTensor.
        :rtype: int
        """
        return self._item_0.numel()

    def img2col(self, k_size: int, stride: int):
        """
        Perform img2col (image to column) operation on both tensors in the RingPair.

        :param k_size: Kernel size for the img2col operation.
        :type k_size: int
        :param stride: Stride size for the img2col operation.
        :type stride: int
        :return: A new RingPair with img2col results and additional info (batch size, output size, channels).
        :rtype: tuple(RingPair, int, int, int)
        """
        res0, _, _, _ = self._item_0.img2col(k_size, stride)
        res1, batch, out_size, channel = self._item_1.img2col(k_size, stride)
        return RingPair(res0, res1), batch, out_size, channel

    def __str__(self):
        """
        Return a string representation of the RingPair.

        :return: A string showing the values of _item_0 and _item_1.
        :rtype: str
        """
        return f"[value0:{self._item_0}\nvalue1:{self._item_1}]"

    def __getitem__(self, item):
        """
        Get the RingTensor in this RingPair by index.

        :param item: The index (0 or 1) to retrieve the corresponding RingTensor.
        :type item: int
        :return: The RingTensor at the specified index.
        :rtype: RingTensor
        :raises IndexError: If the index is not 0 or 1.
        """
        assert item in [0, 1], IndexError("Index out of range")
        return self._item_0 if item == 0 else self._item_1

    def __setitem__(self, key, value):
        """
        Set the RingTensor in this RingPair by index.

        :param key: The index (0 or 1) to set the corresponding RingTensor.
        :type key: int
        :param value: The new RingTensor to set at the specified index.
        :type value: RingTensor
        :raises IndexError: If the index is not 0 or 1.
        """
        assert key in [0, 1], IndexError("Index out of range")
        if key == 0:
            self._item_0 = value
        else:
            self._item_1 = value
