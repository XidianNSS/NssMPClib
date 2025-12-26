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

This method simulates a trusted third party generating parameters locally with provided generation script:

```bash
# Generate all necessary parameters
python scripts/offline_parameter_generation.py
```

#### Method 2: Homomorphic Encryption (HE) - Production Use
For real deployments, use homomorphic encryption to generate parameters without a trusted third party:

```python
from nssmpc import Party2PC, SEMI_HONEST
from nssmpc.protocols.semi_honest_2pc.multiplication import AssMulTriples

# Initialize party
party = Party2PC(0, SEMI_HONEST)
party.online()

# Generate using homomorphic encryption
AssMulTriples.gen_and_save(1000, num_of_party=2, type_of_generation='HE', party=party)
```

```bash

python scripts/offline_parameter_generation.py
```

### Parameter Storage Location
Generated parameters are saved in:

- **32-bit parameters**: `base_path/data/32/`
- **64-bit parameters**: `base_path/data/64/`

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
    "DEBUG_LEVEL": 0,        // Debug verbosity (0-2)
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

from nssmpc.infra.utils.profiling import RuntimeTimer


def expensive_computation(x, y):
    # Your computation here
    return x @ y


# Measure execution time
with RuntimeTimer(tag="Expensive Computation"):
    result, execution_time = expensive_computation(matrix_a, matrix_b)
```
It will print average time and communication stats after execution to stdout.

### Comprehensive Performance Statistics
Get both timing and communication statistics:

```python

from nssmpc.infra.utils.profiling import statistic


def benchmark_function(data):
    # Function to benchmark
    result = secure_computation(data)
    return result


# Run statistics
result, stats = statistic(
    benchmark_function,
    test_data,
    times=10,  # Number of measurement runs
    warmup=5  # Number of warm-up runs (discarded)
)

# OR you can also use the context manager RuntimeTimer to print stats infos directly
with RuntimeTimer(enable_comm_stats=True):
    result, execution_time = expensive_computation(matrix_a, matrix_b)

# It will print time and communication stats after execution to stdout.
```

### Test with Simple Computation
Test the setup with a simple 2-party computation:

**Terminal 1 (Server)**:
```bash
cd tests/primitives/secret_sharing/
python -m unittest test_ass_p0.py
```

**Terminal 2 (Client)**:
```bash
cd tests/primitives/secret_sharing/
python -m unittest test_ass_p1.py
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
   # Terminal 1: python -m unittest test_ass_p0.py
   # Terminal 2: python -m unittest test_ass_p1.py
   ```
3. **Explore applications**: Check out the neural network inference examples
   ```bash
   cd tests/application/neural_network/2pc/
   # Terminal 1: python neural_network_P0.py
   # Terminal 2: python neural_network_P1.py
   ```
4. **Customize configurations**: Modify `configs.json` for your specific use case

## Summary Checklist

Before starting your first MPC computation, ensure you have:

- [ ] ✅ Installed NssMPClib successfully (`pip install -e .`)
- [ ] ✅ Generated cryptographic parameters (using TTP or HE method)
- [ ] ✅ Verified parameter files exist in `base_path/data/`
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