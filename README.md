# NssMPClib - Secure Multi-Party Computation Library
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/XidianNSS/NssMPClib)

## Introduction

NssMPClib is a comprehensive Secure Multi-Party Computation (MPC) library developed by Xidian University NSS Lab. It implements privacy-preserving computation protocols based on both Arithmetic Secret Sharing and Function Secret Sharing.

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
- **Ring-based Computation**: All operations performed on finite rings for cryptographic security

## System Requirements

- **OS**: Linux (required for proper compilation)
- **Python**: 3.10 or higher (recommended: 3.12)
- **PyTorch**: >=2.3.0 (recommended: 2.7.1)
- **Additional**: C++ compiler (gcc/g++), CUDA toolkit (for GPU support)

## Installation

### Step 1: Clone and Install
```bash
git clone https://github.com/XidianNSS/NssMPClib.git
cd NssMPClib
pip install -e .
```

### Step 2: Generate Cryptographic Parameters
Generate essential precomputed parameters for MPC operations:
```bash
python scripts/offline_parameter_generation.py
```

**Note**: Parameters are saved to `~/NssMPClib/data/` (32-bit in `data/32/`, 64-bit in `data/64/`).

## Quick Start: 2-Party Computation Example

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

## Running Built-in Examples

### 1. Arithmetic Secret Sharing (2-Party)
```bash
cd tests/primitives/secret_sharing/
# Terminal 1:
python -m unittest test_ass_server.py
# Terminal 2:
python -m unittest test_ass_client.py
```

### 2. Neural Network Inference (2-Party)
```bash
cd tests/application/neural_network/2pc/
# Terminal 1:
python neural_network_server.py
# Terminal 2:
python neural_network_client.py
```

### 3. Replicated Secret Sharing (3-Party)
```bash
cd tests/primitives/secret_sharing/
# Terminal 1: python -m unittest test_rss_p0.py
# Terminal 2: python -m unittest test_rss_p1.py  
# Terminal 3: python -m unittest test_rss_p2.py
```

## Configuration

Configure the library in `NssMPC/config/configs.json`:
```json
{
    "BIT_LEN": 32,           // Ring size: 32 or 64 bits
    "DEVICE": "cuda",        // Compute device: "cpu" or "cuda"
    "DTYPE": "float",        // Data type: "float" or "int"
    "SCALE_BIT": 8,          // Fixed-point scaling bits
    "DEBUG_LEVEL": 2         // Debug level: 0-Secure, 1-Testing, 2-Development
}
```

**DEBUG_LEVEL Details**:
- **0 (Secure Mode)**: Highest security. All pre-generated keys are destroyed after use, strictly following the One-Time Pad principle.
- **1 (Testing Mode)**: Performance-optimized.  Inputs with the same dimensions reuse the same set of keys, facilitating performance testing and batch operations.
- **2 (Development Mode)**: Convenient for development. Uses a single globally-shared pre-generated key for all operations. **ONLY for non-sensitive development environments**.

**Usage Scenarios**:
- `DEBUG_LEVEL: 0` - Production environments with real sensitive data
- `DEBUG_LEVEL: 1` - Performance testing environments, evaluating performance across different input sizes
- `DEBUG_LEVEL: 2` - Protocol development environments, quickly verifying functional correctness

## Project Structure

```
NssMPClib/
├── NssMPC/                   # Main library source
│   ├── application/          # Privacy-preserving applications
│   ├── config/              # Configuration files
│   ├── infra/               # Infrastructure components
│   ├── primitives/          # Cryptographic primitives
│   ├── protocols/           # MPC protocols
│   └── runtime/             # Runtime coordination
├── data/                     # Precomputed cryptographic parameters
├── tests/                   # Test suite and examples
├── tutorials/               # Detailed tutorials
└── scripts/                 # Utility scripts
```

## Precomputed Cryptographic Parameters

The library uses pre-generated parameters for efficiency. Key types include:

| Parameter Type | Purpose | Typical Use |
|----------------|---------|-------------|
| **AssMulTriples** | Multiplication in Arithmetic Secret Sharing | 2-party computation |
| **BooleanTriples** | AND operations in Boolean Secret Sharing | Secure comparison |
| **RssMulTriples** | Multiplication in Replicated Secret Sharing | 3-party computation |
| **DICFKey** | Distributed Interval Containment Function | Secure comparison |
| **GeLUKey** | Gaussian Error Linear Unit activation | Neural networks |

and so on...

## Tutorials

Detailed tutorials are available in the `tutorials/` directory:

| Tutorial | Description |
|----------|-------------|
| **Tutorial 0** | Library setup and configuration |
| **Tutorial 1** | 2-party secure computation |
| **Tutorial 2** | 3-party secure computation |
| **Tutorial 3** | Privacy-preserving neural network inference |
| **Tutorial 4** | Advanced internal components |

## Best Practices

1. **Separate Processes**: Each party must run in separate terminals
2. **Use Runtime Context**: Always wrap operations in `with PartyRuntime(party):`
3. **Parameter Management**: Generate parameters before first use
4. **Security Selection**: Use DEBUG_LEVEL=0 for production, DEBUG_LEVEL=2 for development

## Troubleshooting

### Common Issues:

1. **"Parameters not found" Error**:
   ```bash
   python scripts/offline_parameter_generation.py
   ```

2. **Port Already in Use**:
   Change base port in `configs.json` or kill existing processes.

3. **CUDA Errors**:
   Set `DEVICE: "cpu"` in config or check CUDA installation.

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citation

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
- **GitHub**: https://github.com/XidianNSS/NssMPClib
- **Issues**: https://github.com/XidianNSS/NssMPClib/issues

## Acknowledgements

Maintained by the Network and System Security (NSS) Laboratory at Xidian University.