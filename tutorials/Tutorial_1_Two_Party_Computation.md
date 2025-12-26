# NssMPC Library Tutorial - 2-Party Computation with Arithmetic Secret Sharing

## Overview
The NssMPC library enables secure multi-party computation using additive secret sharing. This tutorial demonstrates how to perform various arithmetic and comparison operations on secret-shared tensors between two parties (Server and Client) in a semi-honest security model.

## Prerequisites

```python
from nssmpc import Party2PC, PartyRuntime, SEMI_HONEST, SecretTensor
from nssmpc.config.configs import *
```

## Party Initialization

### Party 0 Setup (Separate File)
```python
# Each party must run in a separate file/process
party = Party2PC(0, SEMI_HONEST)  # Party ID 0
party.online()  # Establish connection

with PartyRuntime(party):
    # Local tensor initialization
    x = torch.rand([10, 10]).to(DEVICE)
    y = torch.rand([10, 10]).to(DEVICE)

    # Party 0 is data owner
    # Convert to secret-shared tensors and share with Party 1
    share_x = SecretTensor(tensor=x)
    share_y = SecretTensor(tensor=y)
```

### Party 1 Setup (Separate File)
```python
# Each party must run in a separate file/process
party = Party2PC(1, SEMI_HONEST)  # Party ID 1
party.online()  # Establish connection

with PartyRuntime(party):
    # Receive secret shares from server
    share_x = SecretTensor(src_id=0)  # Shares from party 0
    share_y = SecretTensor(src_id=0)
```

## Important Note: Runtime Context
**All secret sharing operations must be performed within the `PartyRuntime` context manager.** The context manager ensures proper resource management, network communication, and synchronization between parties.

## Basic Operations

### 1. Secret Sharing and Reconstruction

#### Party 0
```python
x = torch.rand(10).to(DEVICE)

with PartyRuntime(party):
    # Server creates shares
    share_x = SecretTensor(tensor=x)

    # Both parties reconstruct (results match on both sides)
    reconstructed_x = share_x.recon().convert_to_real_field()
    # Or specifically restore on party 0:
    reconstructed_x = share_x.restore(target_id=0).convert_to_real_field()
```

#### Party 1

```python
with PartyRuntime(party):
    # Client receives shares
    share_x = SecretTensor(src_id=0)
    
    # Both parties reconstruct (results match on both sides)
    reconstructed_x = share_x.recon().convert_to_real_field()
    # Or specifically restore on party 0:
    reconstructed_x = share_x.restore(target_id=0).convert_to_real_field()
```

### 2. Arithmetic Operations

#### Party 0
```python
with PartyRuntime(party):
    # Addition
    share_z = share_x + share_y
    result = share_z.recon().convert_to_real_field()
    # Element-wise Multiplication
    share_z = share_x * share_y
    result = share_z.recon().convert_to_real_field()
    # Matrix Multiplication (requires Beaver triples without DEBUG_LEVEL 2, will mention later in tutorial 4)
    share_z = share_x @ share_y
    result = share_z.recon().convert_to_real_field()
    
```

#### Party 1

```python
with PartyRuntime(party):
    # Addition
    share_z = share_x + share_y
    result = share_z.recon().convert_to_real_field()
    # Element-wise Multiplication
    share_z = share_x * share_y
    result = share_z.recon().convert_to_real_field()
    # Matrix multiplication (requires Beaver triples without DEBUG_LEVEL 2, will mention later in tutorial 4)
    share_z = share_x @ share_y
    result = share_z.recon().convert_to_real_field()
```

### 3. Comparison Operations (Each Party executes symmetrically)
```python
with PartyRuntime(party):
    # Equality check
    share_z = share_x == share_y
    result = share_z.restore().convert_to_real_field()
    # Greater than or equal
    share_z = share_x >= share_y
    result = share_z.restore().convert_to_real_field()
    # Less than or equal
    share_z = share_x <= share_y
    result = share_z.restore().convert_to_real_field()
    # Strictly greater than
    share_z = share_x > share_y
    result = share_z.restore().convert_to_real_field()
    # Strictly less than
    share_z = share_x < share_y
    result = share_z.restore().convert_to_real_field()
```

### 4. Advanced Mathematical Functions (Each Party executes symmetrically)

```python
with PartyRuntime(party):
    # Division
    share_z = share_x / share_y
    result = share_z.recon().convert_to_real_field()
    # Exponential function
    share_z = share_x.exp()
    result = share_z.recon().convert_to_real_field()
    # Inverse square root
    share_z = share_y.rsqrt()
    result = share_z.recon().convert_to_real_field()
    # Summation along dimension
    share_z = share_z.sum(dim=-1)  # Sum along last dimension
    result = share_z.recon().convert_to_real_field()
```

## Key Points to Remember

1. **Separate Execution**: Server and Client must run in separate processes/files. They cannot be in the same Python script.

2. **Runtime Context**: All secret sharing operations must be performed within the `with PartyRuntime(party):` block. This ensures:
   - Proper network communication
   - Resource cleanup
   - Synchronization between parties

3. **Synchronization**: Both parties must execute corresponding operations in the same order and within their respective runtime contexts.

4. **Beaver Triples**: Matrix multiplication requires pre-generated Beaver triples shared between parties.

5. **Field Conversion**: Use `.convert_to_real_field()` after `.restore()` to get plaintext results.

6. **Tolerance**: Due to fixed-point arithmetic, use appropriate tolerances when comparing results:
   ```python
   torch.allclose(plain_result, mpc_result, atol=1e-2, rtol=1e-2)
   ```

7. **Cleanup**: The `PartyRuntime` context manager automatically handles cleanup when exiting the `with` block. But the
   connections remain open for further operations. You should manually close connections after all computations are
   done:
   ```python
   party.close()
   ```

## Error Handling
- Ensure both parties are online before starting computation
- All operations must be within `PartyRuntime` context
- Verify tensor shapes match for binary operations
- Check that Beaver triples are properly initialized for matrix multiplication
- Use appropriate tolerances for floating-point comparisons

## Performance Considerations
- Pre-generate Beaver triples for better performance
- Batch operations when possible
- Consider computational and communication overhead for large tensors
- Keep operations within the runtime context to minimize context switching overhead

## Complete Example Structure

### Party 0 Script

```python
import torch
import nssmpc
from nssmpc import Party2PC, PartyRuntime, SEMI_HONEST

party = Party2PC(0, SEMI_HONEST)
party.online()

with PartyRuntime(party):
    # All secret sharing operations here
    x = torch.rand([10, 10])
    share_x = nssmpc.SecretTensor(tensor=x)

    # Perform computations symmetrically
    square_x = share_x * share_x

    # reconstruct and convert back to torch.Tensor
    result = share_x.recon(target_id=0).convert_to_real_field()
    print("Result:", result)
```

### Party 1 Script

```python
import nssmpc
from nssmpc import Party2PC, PartyRuntime, SEMI_HONEST

party = Party2PC(1, SEMI_HONEST)
party.online()

with PartyRuntime(party):
    # All secret sharing operations here
    share_x = nssmpc.SecretTensor(src_id=0)

    # Perform computations symmetrically
    square_x = share_x * share_x

    # reconstruct and convert back to torch.Tensor
    result = share_x.recon(target_id=0).convert_to_real_field()
    print("Result:", result)
```

This framework provides a secure way to perform computations on private data without revealing inputs to other parties, using additive secret sharing as the underlying cryptographic primitive.