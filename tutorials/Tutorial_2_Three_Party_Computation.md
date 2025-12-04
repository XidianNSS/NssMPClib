# NssMPC Library Tutorial - 3-Party Computation with Replicated Secret Sharing

## Overview
NssMPC supports both Honest Majority (HONEST_MAJORITY) and Semi-Honest (SEMI_HONEST) security models for three-party replicated secret sharing. In three-party replicated secret sharing, secrets are divided into three parts, with each participant holding one part. No single part contains enough information to recover the secret. To reconstruct the original secret, at least two participants must cooperate and combine their shares.

## Prerequisites

```python
from NssMPC import Party3PC, PartyRuntime, HONEST_MAJORITY, SEMI_HONEST, SecretTensor
from NssMPC.config.configs import DEVICE
import torch
```

## Security Models

### Honest Majority (HONEST_MAJORITY)
- Assumes at least 2 out of 3 parties are honest
- Provides stronger security guarantees
- Can tolerate one malicious party

### Semi-Honest (SEMI_HONEST)
- All parties follow the protocol but may try to learn information from messages
- Weaker security but often more efficient
- Cannot tolerate malicious behavior

## Party Initialization

### Important: Separate Execution
**Each party must run in a completely separate Python process/script.** You cannot initialize multiple parties in the same file. This is crucial for maintaining security isolation and proper network communication.

### Party 0 Setup - Honest Majority Mode (Separate File)
```python
# File: party0_honest.py
# Party 0 - Data Owner/Initiator in Honest Majority mode
party0 = Party3PC(0, HONEST_MAJORITY)

with PartyRuntime(party0):
    party0.online()  # Establish connections with P1 and P2
    
    # All secret sharing operations go here
    # ... rest of party0's code
```

### Party 0 Setup - Semi-Honest Mode (Separate File)
```python
# File: party0_semi.py
# Party 0 - Data Owner/Initiator in Semi-Honest mode
party0 = Party3PC(0, SEMI_HONEST)

with PartyRuntime(party0):
    party0.online()  # Establish connections with P1 and P2
    
    # All secret sharing operations go here
    # ... rest of party0's code
```

### Party 1 Setup (Separate File)
```python
# File: party1.py  
# Party 1 - Computation Participant
# Must use the SAME mode as party0
party1 = Party3PC(1, HONEST_MAJORITY)  # or SEMI_HONEST

with PartyRuntime(party1):
    party1.online()  # Establish connections with P0 and P2
    
    # All secret sharing operations go here
    # ... rest of party1's code
```

### Party 2 Setup (Separate File)
```python
# File: party2.py
# Party 2 - Computation Participant
# Must use the SAME mode as other parties
party2 = Party3PC(2, HONEST_MAJORITY)  # or SEMI_HONEST

with PartyRuntime(party2):
    party2.online()  # Establish connections with P0 and P1
    
    # All secret sharing operations go here
    # ... rest of party2's code
```

**Important**: All three parties must use the **same security mode** (either all `HONEST_MAJORITY` or all `SEMI_HONEST`).

## Important Note: Runtime Context
**All replicated secret sharing operations must be performed within the `PartyRuntime` context manager.** This ensures proper resource management, network communication, and synchronization between all three parties.

## Basic Operations

### 1. Secret Sharing and Reconstruction

#### Party 0: Creating and Sharing Secrets (Both Modes)
```python
# File: party0.py (can be either honest or semi-honest mode)
with PartyRuntime(party0):
    party0.online()
    
    # Create local tensors (only party0 has the original data)
    x = torch.tensor([1.1, 1.1, 1.3], device=DEVICE)
    y = torch.tensor([1.2, 1.1, 2.3], device=DEVICE)
    
    # Convert to secret-shared tensors (automatically shares with P1 and P2)
    share_x = SecretTensor(tensor=x)
    share_y = SecretTensor(tensor=y)
    
    # Reconstruction (requires collaboration from at least 2 parties)
    reconstructed_x = share_x.restore().convert_to_real_field()
    print("Reconstructed x:", reconstructed_x)
```

#### Party 1: Receiving and Working with Shares
```python
# File: party1.py (must match party0's mode)
with PartyRuntime(party1):
    party1.online()
    
    # Receive secret shares (automatically handled by SecretTensor)
    share_x = SecretTensor(src_id=0)  # Shares from party 0
    share_y = SecretTensor(src_id=0)
    
    # Participate in reconstruction
    reconstructed_x = share_x.restore().convert_to_real_field()
    print("Reconstructed x:", reconstructed_x)
```

#### Party 2: Receiving and Working with Shares
```python
# File: party2.py (must match party0's mode)
with PartyRuntime(party2):
    party2.online()
    
    # Receive secret shares
    share_x = SecretTensor(src_id=0)  # Shares from party 0
    share_y = SecretTensor(src_id=0)
    
    # Participate in reconstruction
    reconstructed_x = share_x.restore().convert_to_real_field()
    print("Reconstructed x:", reconstructed_x)
```

### 2. Arithmetic Operations (Same for Both Modes)

**All three parties execute the same operations within their own runtime contexts:**

```python
# In each party's separate file (party0.py, party1.py, party2.py)
with PartyRuntime(party):  # Replace 'party' with party0, party1, or party2
    # Addition
    share_z = share_x + share_y
    result = share_z.restore().convert_to_real_field()
    print("x + y:", result)
    
    # Element-wise Multiplication
    share_z = share_x * share_y
    result = share_z.restore().convert_to_real_field()
    print("x * y:", result)
    
    # Matrix Multiplication (if x and y are matrices)
    share_z = share_x @ share_y
    result = share_z.restore().convert_to_real_field()
    print("x @ y:", result)
```

### 3. Comparison Operations (Same for Both Modes)

```python
# In each party's separate file
with PartyRuntime(party):
    # Equality check
    share_z = share_x == share_y
    result = share_z.restore().convert_to_real_field()
    print("x == y:", result)
    
    # Greater than or equal
    share_z = share_x >= share_y
    result = share_z.restore().convert_to_real_field()
    print("x >= y:", result)
    
    # Less than or equal
    share_z = share_x <= share_y
    result = share_z.restore().convert_to_real_field()
    print("x <= y:", result)
    
    # Strictly greater than
    share_z = share_x > share_y
    result = share_z.restore().convert_to_real_field()
    print("x > y:", result)
    
    # Strictly less than
    share_z = share_x < share_y
    result = share_z.restore().convert_to_real_field()
    print("x < y:", result)
```

### 4. Advanced Mathematical Functions (Same for Both Modes)

```python
# In each party's separate file
with PartyRuntime(party):
    # Division
    share_z = share_x / share_y
    result = share_z.restore().convert_to_real_field()
    print("x / y:", result)
    
    # Exponential function
    share_z = share_x.exp()
    result = share_z.restore().convert_to_real_field()
    print("exp(x):", result)
    
    # Inverse square root
    share_z = share_y.rsqrt()
    result = share_z.restore().convert_to_real_field()
    print("rsqrt(y):", result)
    
    # Summation along dimension
    share_z = share_x.sum(dim=-1)
    result = share_z.restore().convert_to_real_field()
    print("sum(x, dim=-1):", result)
```

## Key Points to Remember

1. **Separate Execution is Mandatory**: Each party must run in its own separate Python process/script file. Never initialize multiple parties in the same file.

2. **Consistent Security Mode**: All three parties must use the same security mode (either all `HONEST_MAJORITY` or all `SEMI_HONEST`).

3. **Runtime Context Required**: All operations must be performed within the `with PartyRuntime(party):` block.

4. **Synchronization**: All three parties must execute corresponding operations in the same order and within their respective runtime contexts.

5. **Field Conversion**: Use `.convert_to_real_field()` after `.restore()` to get plaintext results.

6. **Security Mode Differences**:
   - `HONEST_MAJORITY`: Can tolerate one malicious party, slightly more overhead
   - `SEMI_HONEST`: More efficient, but all parties must follow protocol

## Complete Example Structure

### party0_honest.py (Honest Majority Mode)
```python
# File: party0_honest.py - MUST be run separately
from NssMPC import Party3PC, PartyRuntime, HONEST_MAJORITY, SecretTensor
import torch

# Initialize ONLY party0 in Honest Majority mode
party0 = Party3PC(0, HONEST_MAJORITY)

with PartyRuntime(party0):
    party0.online()
    
    # Only party0 creates the original data
    x = torch.tensor([1.1, 1.1, 1.3])
    share_x = SecretTensor(tensor=x)
    
    # Perform computations
    share_z = share_x + share_x
    result = share_z.restore().convert_to_real_field()
    print("Party 0 (Honest) Result:", result)
```

### party0_semi.py (Semi-Honest Mode)
```python
# File: party0_semi.py - MUST be run separately
from NssMPC import Party3PC, PartyRuntime, SEMI_HONEST, SecretTensor
import torch

# Initialize ONLY party0 in Semi-Honest mode
party0 = Party3PC(0, SEMI_HONEST)

with PartyRuntime(party0):
    party0.online()
    
    # Only party0 creates the original data
    x = torch.tensor([1.1, 1.1, 1.3])
    share_x = SecretTensor(tensor=x)
    
    # Perform computations
    share_z = share_x + share_x
    result = share_z.restore().convert_to_real_field()
    print("Party 0 (Semi-Honest) Result:", result)
```

### party1.py (Matching Mode)
```python
# File: party1.py - MUST be run separately  
# Must match party0's mode (either HONEST_MAJORITY or SEMI_HONEST)
from NssMPC import Party3PC, PartyRuntime, HONEST_MAJORITY, SecretTensor

# Initialize ONLY party1 in SAME mode as party0
party1 = Party3PC(1, HONEST_MAJORITY)  # or SEMI_HONEST

with PartyRuntime(party1):
    party1.online()
    
    # Receive shares from party0
    share_x = SecretTensor(src_id=0)
    
    # Perform SAME computations as party0
    share_z = share_x + share_x
    result = share_z.restore().convert_to_real_field()
    print("Party 1 Result:", result)
```

### party2.py (Matching Mode)
```python
# File: party2.py - MUST be run separately
from NssMPC import Party3PC, PartyRuntime, HONEST_MAJORITY, SecretTensor

# Initialize ONLY party2 in SAME mode as other parties
party2 = Party3PC(2, HONEST_MAJORITY)  # or SEMI_HONEST

with PartyRuntime(party2):
    party2.online()
    
    # Receive shares from party0
    share_x = SecretTensor(src_id=0)
    
    # Perform SAME computations as other parties
    share_z = share_x + share_x
    result = share_z.restore().convert_to_real_field()
    print("Party 2 Result:", result)
```

## Execution Instructions

### For Honest Majority Mode:
1. **Open three separate terminal windows**
2. **In Terminal 1**: Run `python party0_honest.py`
3. **In Terminal 2**: Run `python party1.py` (with `HONEST_MAJORITY`)
4. **In Terminal 3**: Run `python party2.py` (with `HONEST_MAJORITY`)

### For Semi-Honest Mode:
1. **Open three separate terminal windows**
2. **In Terminal 1**: Run `python party0_semi.py`
3. **In Terminal 2**: Run `python party1.py` (with `SEMI_HONEST`)
4. **In Terminal 3**: Run `python party2.py` (with `SEMI_HONEST`)

## Error Handling
- Ensure all three parties use the same security mode
- Each party must be in its own separate process
- All operations must be within `PartyRuntime` context
- Verify network connectivity between all parties

## Performance Considerations
- `SEMI_HONEST` mode is generally more efficient than `HONEST_MAJORITY`
- Communication overhead is higher with three parties
- Batch operations to reduce round complexity
- Honest Majority provides stronger security at the cost of some efficiency

Choose `HONEST_MAJORITY` when you need to tolerate one malicious party, or `SEMI_HONEST` when you prioritize efficiency and trust all parties to follow the protocol.