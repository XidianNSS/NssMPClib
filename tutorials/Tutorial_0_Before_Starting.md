# Tutorial 0: Before Starting NssMPClib

## Overview

Before using NssMPClib for secure multi-party computation, there are several important setup steps you need to complete. This tutorial guides you through the essential preparation work, including generating cryptographic parameters, understanding the configuration system, and learning about debugging utilities.

## Prerequisites

### System Requirements
- **Operating System**: Linux (required for proper compilation)
- **Python**: 3.10 or higher (recommended: Python 3.12)
- **PyTorch**: >=2.3.0 (recommended: PyTorch==2.7.1)

### Installation Verification
First, ensure NssMPClib is properly installed:

```bash
# Clone and install
git clone https://github.com/XidianNSS/NssMPClib.git
cd NssMPClib
pip install -e .

# Verify installation
python -c "import NssMPC; print('NssMPClib successfully imported')"
```

## Step 1: Generate Auxiliary Cryptographic Parameters

### Why Generate Parameters?
Many secure computation protocols require pre-computed cryptographic materials (like Beaver triples) that are generated in an offline phase. These parameters enable efficient online computation while maintaining security.

### Parameter Generation Methods
NssMPClib supports two methods for generating auxiliary parameters:

#### Method 1: Trusted Third Party (TTP) - Recommended for Development
This method simulates a trusted third party generating parameters locally:

```python
from NssMPC.application.neural_network.layers.activation import GeLUKey
from NssMPC.primitives.secret_sharing.function import VSigmaKey, GrottoDICFKey, DICFKey, SigmaDICFKey
from NssMPC.protocols.honest_majority_3pc.msb_with_os import MACKey
from NssMPC.protocols.honest_majority_3pc.multiplication import RssMulTriples
from NssMPC.protocols.honest_majority_3pc.oblivious_select_dpf import VOSKey
from NssMPC.protocols.semi_honest_3pc.truncate import RssTruncAuxParams
from NssMPC.protocols.semi_honest_2pc import Wrap, ReciprocalSqrtKey
from NssMPC.protocols.semi_honest_2pc.b2a import B2AKey
from NssMPC.protocols.semi_honest_2pc.comparison import BooleanTriples
from NssMPC.protocols.semi_honest_2pc.division import DivKey
from NssMPC.protocols.semi_honest_2pc.multiplication import AssMulTriples
from NssMPC.protocols.semi_honest_2pc.tanh import TanhKey

gen_num = 100

AssMulTriples.gen_and_save(gen_num, num_of_party=2, type_of_generation='TTP')
BooleanTriples.gen_and_save(gen_num, num_of_party=2, type_of_generation='TTP')
Wrap.gen_and_save(gen_num)
GrottoDICFKey.gen_and_save(gen_num)
RssMulTriples.gen_and_save(gen_num)
DICFKey.gen_and_save(gen_num)
SigmaDICFKey.gen_and_save(gen_num)
ReciprocalSqrtKey.gen_and_save(gen_num)
DivKey.gen_and_save(gen_num)
GeLUKey.gen_and_save(gen_num)
RssTruncAuxParams.gen_and_save(gen_num)
B2AKey.gen_and_save(gen_num)
TanhKey.gen_and_save(gen_num)
MACKey.gen_and_save(gen_num)

VOSKey.gen_and_save(gen_num, 'VOSKey_0')
VOSKey.gen_and_save(gen_num, 'VOSKey_1')
VOSKey.gen_and_save(gen_num, 'VOSKey_2')

VSigmaKey.gen_and_save(gen_num, 'VSigmaKey_0')
VSigmaKey.gen_and_save(gen_num, 'VSigmaKey_1')
VSigmaKey.gen_and_save(gen_num, 'VSigmaKey_2')

B2AKey.gen_and_save(gen_num, 'B2AKey_0')
B2AKey.gen_and_save(gen_num, 'B2AKey_1')
B2AKey.gen_and_save(gen_num, 'B2AKey_2')
```

#### Method 2: Homomorphic Encryption (HE) - Production Use
For real deployments, use homomorphic encryption to generate parameters without a trusted third party:

```python
from NssMPC.infra.mpc.party import SemiHonestCS
from NssMPC.protocols.semi_honest_2pc import AssMulTriples

# Initialize party
party = SemiHonestCS(type='server')
party.online()

# Generate using homomorphic encryption
AssMulTriples.gen_and_save(1000, num_of_party=2, type_of_generation='HE', party=party)
```

#### Method 3: Using the Provided Script (Easiest)
The easiest way is to use the provided generation script:

```bash
# Generate all necessary parameters
python scripts/offline_parameter_generation.py
```

### Parameter Storage Location
Generated parameters are saved in:
- **32-bit parameters**: `~/NssMPClib/data/32/`
- **64-bit parameters**: `~/NssMPClib/data/64/`

Each parameter type has its own subdirectory with separate files for each party.

### Parameter Types and Their Uses

| Parameter Type | Purpose | Used In |
|----------------|---------|---------|
| **AssMulTriples** | Multiplication in Arithmetic Secret Sharing | 2-party and 3-party ASS |
| **BooleanTriples** | AND operations in Boolean Secret Sharing | 2-party Boolean sharing |
| **RssMulTriples** | Multiplication in Replicated Secret Sharing | 3-party RSS |
| **DICFKey** | Distributed Interval Containment Function | Secure comparison |
| **GrottoDICFKey** | Faster DICF implementation | Optimized comparison |
| **SigmaDICFKey** | DReLU function implementation | Comparison with different security |
| **GeLUKey** | Gaussian Error Linear Unit activation | Neural networks |
| **ReciprocalSqrtKey** | Reciprocal square root operations | Normalization |
| **DivKey** | Secure division operations | Division |
| **TanhKey** | Tanh activation function | Neural networks |
| **MACKey** | Message Authentication Code keys | Malicious-secure verification |
| **VOSKey** | Oblivious selection | Malicious-secure protocols |
| **VSigmaKey** | Verifiable sigma protocol keys | Malicious-secure comparison |
| **Wrap** | Truncation operations | Precision management |
| **RssTruncAuxParams** | Truncation in RSS | 3-party truncation |
| **B2AKey** | Boolean-to-Arithmetic conversion | Type conversion |

**Note**: Matrix multiplication requires matrix-specific Beaver triples, which are generated on-demand based on the matrix sizes involved in your computation. These will be covered in later tutorials.

## Step 2: Configuration File Setup

### Configuration File Location
When you first run NssMPClib, it automatically generates a configuration file at:
- **Configuration file**: `NssMPC/config/configs.json`

### Understanding Configuration Options
The configuration file controls the behavior of NssMPClib. Here are the key options:

```json
{
    "BIT_LEN": 32,           // Ring size: 32 or 64 bits (affects security and performance)
    "DEBUG_LEVEL": 0,              // Debug verbosity (0-2)
    "DTYPE": "float",        // Data type: "float" or "int"
    "SCALE_BIT": 8,          // Fixed-point scaling bits for fractional numbers
    "DEVICE": "cuda",        // Computation device: "cpu" or "cuda" (GPU)
    ... other configurations ...
}
```

### Important Configuration Notes

1. **BIT_LEN**: Determines the ring size for cryptographic operations:
   - `32`: Faster, lower security (suitable for development)
   - `64`: Slower, higher security (recommended for production)

2. **DEVICE**: 
   - Set to `"cuda"` if you have NVIDIA GPU and CUDA installed
   - Set to `"cpu"` for CPU-only computation

3. **SCALE_BIT**: Important for fixed-point arithmetic:
   - Determines precision for fractional numbers
   - Higher values mean more precision but slower computation
   - Default 8 is sufficient for most neural network applications

## Step 3: Debugging and Performance Utilities

### Time Measurement
Measure execution time of any function:

```python
from NssMPC.infra.utils.debug_utils import get_time

def expensive_computation(x, y):
    # Your computation here
    return x @ y

# Measure execution time
result, execution_time = get_time(expensive_computation, matrix_a, matrix_b)
```
It will print average time and communication stats after execution to stdout.

### Comprehensive Performance Statistics
Get both timing and communication statistics:

```python
from NssMPC.infra.utils.debug_utils import statistic

def benchmark_function(data):
    # Function to benchmark
    result = secure_computation(data)
    return result

# Run statistics
result, stats = statistic(
    benchmark_function, 
    test_data,
    times=10,     # Number of measurement runs
    warmup=5      # Number of warm-up runs (discarded)
)

```
It will print average time and communication stats after execution to stdout.

## Step 4: Quick Verification

### Verify Installation and Parameters
Create a simple verification script `verify_setup.py`:

```python
import os
import sys
import torch
from pathlib import Path

def check_installation():
    """Verify NssMPClib is properly installed"""
    try:
        import NssMPC
        print("✅ NssMPClib imported successfully")
        
        # Check PyTorch version
        print(f"✅ PyTorch version: {torch.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Failed to import NssMPClib: {e}")
        return False

def check_parameters():
    """Verify parameter files exist"""
    home = str(Path.home())
    param_dirs = [
        f"{home}/NssMPClib/data/32/",
        f"{home}/NssMPClib/data/64/"
    ]
    
    all_exist = True
    for param_dir in param_dirs:
        if os.path.exists(param_dir):
            param_count = len(list(Path(param_dir).rglob("*.pkl")))
            print(f"✅ Found {param_count} parameter files in {param_dir}")
        else:
            print(f"❌ Parameter directory not found: {param_dir}")
            all_exist = False
    
    return all_exist

def check_config():
    """Verify configuration file"""
    config_path = "NssMPC/config/configs.json"
    if os.path.exists(config_path):
        print(f"✅ Configuration file found: {config_path}")
        return True
    else:
        print(f"❌ Configuration file not found: {config_path}")
        return False

def main():
    print("=" * 50)
    print("NssMPClib Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Installation", check_installation),
        ("Parameters", check_parameters),
        ("Configuration", check_config)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\nChecking {name}...")
        results.append(check_func())
    
    print("\n" + "=" * 50)
    print("Verification Summary:")
    print("=" * 50)
    
    for i, (name, _) in enumerate(checks):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{name}: {status}")
    
    if all(results):
        print("\n🎉 All checks passed! You're ready to use NssMPClib.")
    else:
        print("\n⚠️ Some checks failed. Please review the setup instructions.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Run the verification:
```bash
python verify_setup.py
```

### Test with Simple Computation
Test the setup with a simple 2-party computation:

**Terminal 1 (Server)**:
```bash
cd tests/primitives/secret_sharing/
python -m unittest test_ass_server.py
```

**Terminal 2 (Client)**:
```bash
cd tests/primitives/secret_sharing/
python -m unittest test_ass_client.py
```

## Step 5: Common Setup Issues and Solutions

### Issue 1: "Parameters not found" Error
**Symptoms**: Runtime errors about missing parameter files
**Solution**: Generate parameters using the script:
```bash
python scripts/offline_parameter_generation.py
```

### Issue 2: CUDA/CUDNN Errors
**Symptoms**: Errors about CUDA initialization or missing libraries
**Solution**:
1. Check CUDA installation: `nvcc --version`
2. If CUDA not available, set device to CPU in config:
   ```json
   {"DEVICE": "cpu"}
   ```
3. Or set environment variable: `export CUDA_VISIBLE_DEVICES=""`

### Issue 3: Port Already in Use
**Symptoms**: Network connection errors during party initialization
**Solution**:
1. Change the base port in config.json
2. Or kill existing processes:
   ```bash
   sudo lsof -t -i:12345 | xargs kill -9
   ```

## Step 6: Next Steps

After completing this setup, you're ready to:

1. **Learn the basics**: Proceed to Tutorial 1 to learn about RingTensor operations and basic MPC concepts
2. **Try examples**: Run the test scripts in `tests/` directory
   ```bash
   # 2-party arithmetic sharing
   cd tests/primitives/secret_sharing/
   # Terminal 1: python -m unittest test_ass_server.py
   # Terminal 2: python -m unittest test_ass_client.py
   ```
3. **Explore applications**: Check out the neural network inference examples
   ```bash
   cd tests/application/neural_network/2pc/
   # Terminal 1: python neural_network_server.py
   # Terminal 2: python neural_network_client.py
   ```
4. **Customize configurations**: Modify `configs.json` for your specific use case

## Summary Checklist

Before starting your first MPC computation, ensure you have:

- [ ] ✅ Installed NssMPClib successfully (`pip install -e .`)
- [ ] ✅ Generated cryptographic parameters (using TTP or HE method)
- [ ] ✅ Verified parameter files exist in `~/NssMPClib/data/`
- [ ] ✅ Configuration file created (`NssMPC/config/configs.json`)
- [ ] ✅ Set appropriate configuration (BIT_LEN, DEVICE, etc.)
- [ ] ✅ Tested basic 2-party computation
- [ ] ✅ Read and understood the security implications of your chosen parameters

## Security Considerations

1. **Parameter Generation**:
   - TTP method is for **development only**
   - Use HE method for **production deployments**
   - Never share parameter files between different security domains

2. **Configuration**:
   - Higher BIT_LEN (64) provides better security
   - Ensure proper network isolation in production
   - Regularly rotate cryptographic parameters

3. **Performance vs Security**:
   - 32-bit: Faster, less secure (development/testing)
   - 64-bit: Slower, more secure (production)

## Getting Help

If you encounter issues:

1. **Check the documentation**: Review README.md and tutorials
2. **Examine logs**: Enable DEBUG mode in configuration
3. **Run verification**: Use the verification script above
4. **Check GitHub issues**: https://github.com/XidianNSS/NssMPClib/issues
5. **Contact support**: xidiannss@gmail.com

You're now ready to start using NssMPClib for secure multi-party computation! Continue to the next tutorials to learn how to implement specific secure computation tasks.