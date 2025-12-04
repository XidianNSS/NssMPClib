# NssMPClib - Secure Multi-Party Computation Library
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/XidianNSS/NssMPClib)

## Introduction

NssMPClib is a comprehensive Secure Multi-Party Computation (MPC) library developed by Xidian University NSS Lab. It implements privacy-preserving computation protocols based on both Arithmetic Secret Sharing and Function Secret Sharing, with practical applications in privacy-preserving machine learning, particularly neural network inference.

## Key Features

- **Multiple Security Models**: Supports both Semi-Honest and Honest-Majority security assumptions
- **Flexible Party Configurations**: 2-party and 3-party computation setups
- **Multiple Secret Sharing Schemes**:
  - Additive Secret Sharing (2-party)
  - Replicated Secret Sharing (3-party)
- **Function Secret Sharing (FSS)** implementations:
  - Distributed Point Function (DPF)
  - Distributed Comparison Function (DCF)
  - Distributed Interval Containment Function (DICF) with multiple variants (Standard, GROTTO, SIGMA)
- **Privacy-Preserving Neural Network Inference**: Support for secure model evaluation
- **Precomputed Parameter System**: Efficient Beaver triple generation via homomorphic encryption
- **Ring-based Computation**: All operations performed on finite rings for cryptographic security
- **Comprehensive Precomputed Parameters**: Includes pre-generated cryptographic parameters for common operations

## System Requirements

### Operating System
- **Linux** (required for proper compilation and execution of csprng and C++ extensions)
- Tested on: Ubuntu 20.04+, CentOS 7+

### Python Dependencies
- **Python**: 3.10 or higher (tested with Python 3.12)
- **PyTorch**: >=2.5.0 (recommended: PyTorch==2.7.0)
- **Additional**: C++ compiler (gcc/g++), CUDA toolkit (for GPU support)

### Build Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential gcc g++ python3-dev

# For CUDA support
sudo apt-get install nvidia-cuda-toolkit
```

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/XidianNSS/NssMPClib.git
cd NssMPClib
```

### Step 2: Build and Install csprng & Install NssMPClib
```bash
# Install the main library
pip install -e .
```

### Step 3: Generate Offline Parameters (Optional)
```bash
# Generate precomputed cryptographic parameters
python scripts/offline_parameter_generation.py
```


### Troubleshooting Common Issues

1. **csprng compilation fails**:
   ```bash
   # Ensure you have the correct build tools
   sudo apt-get install build-essential python3-dev
   ```

2. **CUDA-related errors**:
   ```bash
   # Check CUDA installation
   nvcc --version
   # If CUDA not available, use CPU-only mode
   export CUDA_VISIBLE_DEVICES=""
   ```

## Quick Start

### Basic Setup (2-Party Computation)

**Server (Party 0) - `server.py`**:
```python
from NssMPC import Party2PC, PartyRuntime, SEMI_HONEST, SecretTensor
import torch

server = Party2PC(0, SEMI_HONEST)
with PartyRuntime(server):
    server.online()
    x = torch.rand([10, 10])
    share_x = SecretTensor(tensor=x)
    result = share_x.restore().convert_to_real_field()
    print("Server result:", result)
```

**Client (Party 1) - `client.py`**:
```python
from NssMPC import Party2PC, PartyRuntime, SEMI_HONEST, SecretTensor

client = Party2PC(1, SEMI_HONEST)
with PartyRuntime(client):
    client.online()
    share_x = SecretTensor(src_id=0)
    result = share_x.restore().convert_to_real_field()
    print("Client result:", result)
```

**Execution**:
```bash
# Terminal 1: Start server
python server.py

# Terminal 2: Start client (in separate terminal)
python client.py
```

### Running Pre-Built Examples

The repository includes comprehensive test examples:

#### 1. Arithmetic Secret Sharing Tests
```bash
# 2-Party test
cd tests/primitives/secret_sharing/
python -m unittest test_ass_server.py  # Terminal 1
python -m unittest test_ass_client.py  # Terminal 2

# 3-Party test
python -m unittest test_rss_p0.py  # Terminal 1
python -m unittest test_rss_p1.py  # Terminal 2  
python -m unittest test_rss_p2.py  # Terminal 3
```

#### 2. Neural Network Inference Tests
```bash
# 2-Party NN inference
cd tests/application/neural_network/2pc/
python neural_network_server.py  # Terminal 1
python neural_network_client.py  # Terminal 2

# 3-Party NN inference
cd tests/application/neural_network/3pc/
python P0.py  # Terminal 1
python P1.py  # Terminal 2
python P2.py  # Terminal 3
```

#### 3. Function Secret Sharing Tests
```bash
cd tests/primitives/secret_sharing/
python -m unittest test_fss.py
```

## Project Structure

```
NssMPClib/
├── csprng/                   # Custom cryptographic random number generator
│   ├── torchcsprng/csrc/     # C++/CUDA source code for PRNG
│   └── setup.py              # Build configuration
├── data/                     # Data and precomputed parameters
│   ├── 32/                   # 32-bit precomputed parameters
│   ├── 64/                   # 64-bit precomputed parameters
│   ├── AlexNet/              # AlexNet model implementation
│   ├── NN/                   # Neural network models and datasets
│   └── ResNet/               # ResNet model implementation
├── NssMPC/                   # Main library source
│   ├── application/          # Privacy-preserving applications
│   │   └── neural_network/  # Secure neural network inference
│   ├── config/              # Configuration files
│   ├── infra/               # Infrastructure components
│   │   ├── mpc/             # MPC runtime management
│   │   ├── prg/             # Pseudo-random generators
│   │   ├── tensor/          # RingTensor implementation
│   │   └── utils/           # Utility functions
│   ├── primitives/          # Cryptographic primitives
│   │   ├── homomorphic_encryption/  # HE implementations
│   │   ├── oblivious_transfer/      # OT protocols
│   │   └── secret_sharing/          # Secret sharing schemes
│   ├── protocols/           # MPC protocols
│   │   ├── honest_majority_3pc/    # 3-party honest majority
│   │   ├── semi_honest_2pc/        # 2-party semi-honest
│   │   └── semi_honest_3pc/        # 3-party semi-honest
│   └── runtime/             # Runtime coordination
├── tests/                   # Test suite
│   ├── application/         # Application tests
│   ├── infra/               # Infrastructure tests
│   ├── primitives/          # Primitive tests
│   └── protocols/           # Protocol tests
├── tutorials/               # markdown tutorials
├── scripts/                 # Utility scripts
├── setup.py                 # Package installation
└── pyproject.toml           # Build system configuration
```

## Data and Precomputed Parameters

The library includes precomputed cryptographic parameters for common operations:

### Available Parameters
- **32-bit and 64-bit versions** in `data/32/` and `data/64/`
- **Parameter types include**:
  - `AssMulTriples`: Arithmetic multiplication triples
  - `BooleanTriples`: Boolean triples for comparison
  - `DICFKey`: DICF function keys
  - `RssMulTriples`: Replicated secret sharing triples
  - `SigmaDICFKey`: SIGMA DICF keys
  - And more for specific operations

### Using Precomputed Parameters
```python
from NssMPC.config.configs import *
# The library automatically loads parameters based on FIELD_SIZE setting
```

## Tutorials

The library includes comprehensive tutorials:

| Tutorial | File | Description |
|----------|------|-------------|
| **Tutorial 0** | `tutorials/Tutorial_0_Before_Starting.md` | Library configuration and setup |
| **Tutorial 1** | `tutorials/Tutorial_1_Two_Party_Computation.md` | 2-party secure computation |
| **Tutorial 2** | `tutorials/Tutorial_2_Three_Party_Computation.md` | 3-party secure computation |
| **Tutorial 3** | `tutorials/Tutorial_3_Neural_Network_Inference.md` | Privacy-preserving NN inference |
| **Tutorial 4** | `tutorials/Tutorial_4_Other_Internal_Components.md` | Advanced components |

To run tutorials:
```bash
jupyter notebook tutorials/
```

## Advanced Usage

### 3-Party Neural Network Inference
```bash
# Open three separate terminals

# Terminal 1 - Party 0 (Model Owner)
cd tests/application/neural_network/3pc/
python P0.py

# Terminal 2 - Party 1
python P1.py

# Terminal 3 - Party 2  
python P2.py
```

## Configuration

Key configuration options in `NssMPC/config/configs.json`:

```
# Security parameters
"BIT_LEN": 32,       # Ring size: 32 or 64 bits
"DEVICE": "cuda",    # Debug verbosity (0-2)
# Precision settings
"DTYPE": "float",    # Data type: float or int
"SCALE_BIT": 8,      # Fixed-point scaling bits
# Device configuration
"DEVICE": "cuda",    # 'cpu' or 'cuda' for GPU support
```

## Best Practices

1. **Separate Processes**: Each party must run in separate terminals/processes
2. **Runtime Context**: Always use `with PartyRuntime(party):` blocks
3. **Clean Ports**: Ensure network ports are free before starting
4. **Parameter Management**: Use precomputed parameters for production

## Performance Tips

- **Use precomputed parameters** from `data/` directory
- **Batch operations** when possible to reduce overhead
- **Choose appropriate ring size** (32-bit vs 64-bit) for your use case
- **Enable GPU support** for large computations by setting `DEVICE = 'cuda'`

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citing NssMPClib

If you use NssMPClib in your research, please cite:
```
@software{nssmpclib,
  title = {NssMPClib: Secure Multi-Party Computation Library},
  author = {Xidian University NSS Lab},
  year = {2024},
  url = {https://github.com/XidianNSS/NssMPClib}
}
```

## License

NssMPClib is released under the MIT License. See the LICENSE file for details.

## Contact

- **Email**: xidiannss@gmail.com
- **Website**: https://www.xidiannss.com
- **GitHub**: https://github.com/XidianNSS/NssMPClib
- **Issues**: https://github.com/XidianNSS/NssMPClib/issues

## Acknowledgements

This project is maintained by the Network and System Security (NSS) Laboratory at Xidian University. We thank all contributors and users for their support and feedback.

---

**Note**: This library requires Linux for proper compilation and execution due to dependencies in the torch.distributed component. Windows and macOS are not officially supported.