# NssMPC Library Developer Reference

## Overview
NssMPC is a comprehensive multi-party computation library that provides secure computation capabilities through various cryptographic primitives. This document serves as a reference for developers who want to understand the core components and extend the library's functionality.

## Core Data Structure: RingTensor

### Basics
`RingTensor` is the fundamental data structure that represents tensors over finite rings, analogous to PyTorch tensors but with cryptographic properties.

```python
from nssmpc.infra.tensor import RingTensor
import torch

# Convert between torch tensors and RingTensors
x = torch.tensor([1.0, 2.0, 3.0])
x_on_ring = RingTensor.convert_to_ring(x)
x_real = x_on_ring.convert_to_real_field()

# Supported data types: int64, int32, float64, float32
x_int64 = torch.tensor([1, 2, 3], dtype=torch.int64)
x_int64_on_ring = RingTensor.convert_to_ring(x_int64)
```

### Operations
RingTensor supports most tensor operations with automatic conversion to ring arithmetic:

```python
# Arithmetic operations
x = RingTensor.convert_to_ring(torch.tensor([1.0, 2.0, 3.0]))
y = RingTensor.convert_to_ring(torch.tensor([2.0]))

addition = x + y
subtraction = x - y
multiplication = x * y
matrix_mul = x @ y.unsqueeze(1)

# Comparisons (return boolean tensors)
less_than = x < y
greater_equal = x >= y
equal = x == y

# Tensor manipulations
concatenated = RingTensor.cat((x, y))
stacked = RingTensor.stack((x, y))
reshaped = x.reshape(3, 1)

# Special functions
condition = x > y
result = RingTensor.where(condition, x, y)
random_tensor = RingTensor.random(shape=(2, 3), dtype='int', bounds=(0, 10))
range_tensor = RingTensor.arange(0, 10, 2, dtype='int')
```

---

## Parameter System for Precomputed Values

### Creating Custom Parameters
The Parameter system manages precomputed values (like Beaver triples) that are generated offline and consumed online.

```python
from nssmpc.infra.mpc.aux_parameter import Parameter
from nssmpc.infra.tensor import RingTensor
from nssmpc.primitives.secret_sharing import AdditiveSecretSharing


class CustomParameter(Parameter):
    def __init__(self, a=None, b=None, c=None):
        self.attr_a = a
        self.attr_b = b
        self.attr_c = c

    @staticmethod
    def gen(num, param0):
        """Generate num instances of this parameter type"""
        a = RingTensor.random((num,))
        b = RingTensor.convert_to_ring(param0).repeat((num,))
        c = a + b

        # Secret share the values
        a_0, a_1 = AdditiveSecretSharing.share(a)
        b_0, b_1 = AdditiveSecretSharing.share(b)
        c_0, c_1 = AdditiveSecretSharing.share(c)

        return CustomParameter(a_0, b_0, c_0), CustomParameter(a_1, b_1, c_1)
```

### Using Parameters

> **Chapters in progress...**

This reference provides the essential patterns and components for extending NssMPC. For specific use cases, refer to the existing implementations as templates and follow the established patterns for consistency and security