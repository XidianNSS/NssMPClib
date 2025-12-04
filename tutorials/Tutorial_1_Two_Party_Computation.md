# NssMPC Library Tutorial - 2-Party Computation with Arithmetic Secret Sharing

## Overview
The NssMPC library enables secure multi-party computation using additive secret sharing. This tutorial demonstrates how to perform various arithmetic and comparison operations on secret-shared tensors between two parties (Server and Client) in a semi-honest security model.

## Prerequisites

```python
import NssMPC
from NssMPC import Party2PC, PartyRuntime, SEMI_HONEST
from NssMPC.config.configs import *
```

## Party Initialization

### Server Setup (Separate File)
```python
# Server must run in a separate file/process
server = Party2PC(0, SEMI_HONEST)  # Party ID 0
server.online()  # Establish connection

with PartyRuntime(server):
    # Local tensor initialization
    x = torch.rand([10, 10]).to(DEVICE)
    y = torch.rand([10, 10]).to(DEVICE)
    
    # Convert to secret-shared tensors
    share_x = NssMPC.SecretTensor(tensor=x)
    share_y = NssMPC.SecretTensor(tensor=y)
```

### Client Setup (Separate File)
```python
# Client must run in a separate file/process
client = Party2PC(1, SEMI_HONEST)  # Party ID 1
client.online()  # Establish connection

with PartyRuntime(client):
    # Receive secret shares from server
    share_x = NssMPC.SecretTensor(src_id=0)  # Shares from party 0
    share_y = NssMPC.SecretTensor(src_id=0)
```

## Important Note: Runtime Context
**All secret sharing operations must be performed within the `PartyRuntime` context manager.** The context manager ensures proper resource management, network communication, and synchronization between parties.

## Basic Operations

### 1. Secret Sharing and Reconstruction
```python
with PartyRuntime(server):
    # Server creates shares
    share_x = NssMPC.SecretTensor(tensor=x)
    
    # Both parties reconstruct (results match on both sides)
    reconstructed_x = share_x.restore().convert_to_real_field()
```

### 2. Arithmetic Operations
```python
with PartyRuntime(server):
    # Addition
    share_z = share_x + share_y
    result = share_z.restore().convert_to_real_field()

    # Element-wise Multiplication
    share_z = share_x * share_y
    result = share_z.restore().convert_to_real_field()

    # Matrix Multiplication (requires Beaver triples)
    # Server prepares triples
    triples = MatmulTriples.gen(1, x.shape, y.shape)
    server.providers[MatmulTriples].param = [triples[0]]
    server.send(triples[1])
    server.providers[MatmulTriples].load_mat_beaver()

with PartyRuntime(client):
    # Client receives triples
    client.providers[MatmulTriples].param = [client.recv()]
    client.providers[MatmulTriples].load_mat_beaver()

    # Perform matrix multiplication
    share_z = share_x @ share_y
    result = share_z.restore().convert_to_real_field()
```

### 3. Comparison Operations
```python
with PartyRuntime(server):
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

### 4. Advanced Mathematical Functions
```python
with PartyRuntime(server):
    # Division
    share_z = share_x / share_y
    result = share_z.restore().convert_to_real_field()

    # Exponential function
    share_z = share_x.exp()
    result = share_z.restore().convert_to_real_field()

    # Inverse square root
    share_z = share_y.rsqrt()
    result = share_z.restore().convert_to_real_field()

    # Summation along dimension
    share_z = share_x.exp()
    share_z = share_z.sum(dim=-1)  # Sum along last dimension
    result = share_z.restore().convert_to_real_field()
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

7. **Cleanup**: The `PartyRuntime` context manager automatically handles cleanup when exiting the `with` block. You can also manually close connections:
   ```python
   server.close()  # or client.close()
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

### Server Script
```python
import NssMPC
from NssMPC import Party2PC, PartyRuntime, SEMI_HONEST

server = Party2PC(0, SEMI_HONEST)
server.online()

with PartyRuntime(server):
    # All secret sharing operations here
    x = torch.rand([10, 10])
    share_x = NssMPC.SecretTensor(tensor=x)
    
    # Perform computations
    result = share_x.restore().convert_to_real_field()
    print("Result:", result)
```

### Client Script
```python
import NssMPC
from NssMPC import Party2PC, PartyRuntime, SEMI_HONEST

client = Party2PC(1, SEMI_HONEST)
client.online()

with PartyRuntime(client):
    # All secret sharing operations here
    share_x = NssMPC.SecretTensor(src_id=0)
    
    # Perform computations
    result = share_x.restore().convert_to_real_field()
    print("Result:", result)
```

This framework provides a secure way to perform computations on private data without revealing inputs to other parties, using additive secret sharing as the underlying cryptographic primitive.