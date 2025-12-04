# NssMPC Library Developer Reference

## Overview
NssMPC is a comprehensive multi-party computation library that provides secure computation capabilities through various cryptographic primitives. This document serves as a reference for developers who want to understand the core components and extend the library's functionality.

## Core Data Structure: RingTensor

### Basics
`RingTensor` is the fundamental data structure that represents tensors over finite rings, analogous to PyTorch tensors but with cryptographic properties.

```python
from NssMPC import RingTensor
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
from NssMPC.crypto.aux_parameter import Parameter
from NssMPC import RingTensor
from NssMPC.primitives import AdditiveSecretSharing

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
```python
# Generate and save parameters offline
CustomParameter.gen_and_save(100, 'custom_param', 72)

# Load and use parameters online
from NssMPC.infra.mpc.aux_parameter import ParamProvider
from NssMPC.infra.mpc.party import SemiHonestCS

party = SemiHonestCS('client')
party.append_provider(ParamProvider(CustomParameter, 'custom_param'))
party.online()

# Retrieve parameters when needed
my_param = party.get_param(CustomParameter, 3)
```

---

## Beaver Triple Generation

### Homomorphic Encryption-based Generation
Generate Beaver triples without a trusted third party:

```python
from NssMPC.infra.mpc.party import SemiHonestCS
from NssMPC.protocols.semi_honest_2pc import AssMulTriples, BooleanTriples

# Setup parties
server = SemiHonestCS(type='server')
client = SemiHonestCS(type='client')

for party in [server, client]:
    party.set_multiplication_provider()
    party.set_comparison_provider()
    party.set_nonlinear_operation_provider()
    party.online()

# Generate multiplication triples
AssMulTriples.gen_and_save(10, 2, type_of_generation='HE', party=server)
AssMulTriples.gen_and_save(10, 2, type_of_generation='HE', party=client)

# Generate MSB (Most Significant Bit) triples for comparison
BooleanTriples.gen_and_save(10, 2, type_of_generation='HE', party=server)
BooleanTriples.gen_and_save(10, 2, type_of_generation='HE', party=client)
```

---

## Function Secret Sharing (FSS)

### DPF (Distributed Point Function)
Evaluates a point function: $f_{\alpha,\beta}(x) = \beta$ if $x = \alpha$, else 0

```python
from NssMPC.primitives import DPF, DPFKey
from NssMPC import RingTensor

# Offline: Generate keys
alpha = RingTensor.convert_to_ring(5)
beta = RingTensor.convert_to_ring(1)
key0, key1 = DPFKey.gen(num_of_keys=10, alpha=alpha, beta=beta)

# Online: Evaluate
x = RingTensor.convert_to_ring([1, 2, 3, 4, 5])
res0 = DPF.eval(x=x, keys=key0, party_id=0)
res1 = DPF.eval(x=x, keys=key1, party_id=1)
result = res0 + res1  # Should be 1 at position 5

# Different ring sizes
alpha_small = alpha.convert_to_range(bit_len=8)
x_small = x.convert_to_range(bit_len=8)
key0_small, key1_small = DPFKey.gen(num_of_keys=10, alpha=alpha_small, beta=beta)
```

### DCF (Distributed Comparison Function)
Evaluates: $f_{\alpha,\beta}(x) = \beta$ if $x < \alpha$, else 0

```python
from NssMPC.primitives import DCF
key0, key1 = DCF.gen(num_of_keys=10, alpha=alpha, beta=beta)
res0 = DCF.eval(x=x, keys=key0, party_id=0)
res1 = DCF.eval(x=x, keys=key1, party_id=1)
result = res0 + res1
```

### DICF (Distributed Interval Containment Function)
Three implementations available for different performance/security trade-offs:

#### 1. Standard DICF (2021 FSS Paper)
```python
from NssMPC.primitives import DICF, DICFKey
down_bound = RingTensor(3)
upper_bound = RingTensor(7)
key0, key1 = DICFKey.gen(num_of_keys=10, down_bound=down_bound, upper_bound=upper_bound)

x = RingTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_shift = x + key0.r_in.reshape(x.shape) + key1.r_in.reshape(x.shape)

res0 = DICF.eval(x_shift=x_shift, keys=key0, party_id=0, 
                 down_bound=down_bound, upper_bound=upper_bound)
res1 = DICF.eval(x_shift=x_shift, keys=key1, party_id=1,
                 down_bound=down_bound, upper_bound=upper_bound)
result = res0 + res1  # 1 for values 3-7, 0 otherwise
```

#### 2. GROTTO Implementation
Faster implementation using prefix sums and DPF. Returns Boolean secret shares.

```python
from NssMPC.primitives import GrottoDICF, GrottoDICFKey
key0, key1 = GrottoDICFKey.gen(num_of_keys=10, beta=RingTensor(1))

x_shift = key0.r_in.reshape(x.shape) + key1.r_in.reshape(x.shape) - x
res0 = GrottoDICF.eval(x_shift=x_shift, key=key0, party_id=0,
                       down_bound=down_bound, upper_bound=upper_bound)
res1 = GrottoDICF.eval(x_shift=x_shift, key=key1, party_id=1,
                       down_bound=down_bound, upper_bound=upper_bound)
result = res0 ^ res1  # XOR for Boolean shares
```

#### 3. SIGMA Implementation
Implements DReLU function: $f(x) = 1$ if $x \geq 0$, else 0

```python
from NssMPC.primitives import SigmaDICF, SigmaDICFKey
key0, key1 = SigmaDICFKey.gen(num_of_keys=10)

x = RingTensor([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
x_shift = key0.r_in.reshape(x.shape) + key1.r_in.reshape(x.shape) + x

res0 = SigmaDICF.eval(x_shift=x_shift, key=key0, party_id=0)
res1 = SigmaDICF.eval(x_shift=x_shift, key=key1, party_id=1)
result = res0 ^ res1  # Boolean shares

# Different ring sizes
key0_small, key1_small = SigmaDICFKey.gen(num_of_keys=10, bit_len=8)
x_shift_small = x_shift.convert_to_range(bit_len=8)
```

---

This reference provides the essential patterns and components for extending NssMPC. For specific use cases, refer to the existing implementations as templates and follow the established patterns for consistency and security