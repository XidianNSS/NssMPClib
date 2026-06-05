# NssMPClib - A General-Purpose Secure Multi-Party Computation Library Based on PyTorch
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/XidianNSS/NssMPClib)

## Introduction

NssMPClib is a secure multi-party computation (MPC) library designed specifically for machine learning, offering
familiar PyTorch-style APIs that make privacy-preserving machine learning development as straightforward as regular
PyTorch programming.

It implements diverse privacy-preserving computation protocols based on both Arithmetic Secret Sharing and Function
Secret Sharing.
## Key Features

- **PyTorch Integration**: Leverages PyTorch tensor operations for ease of use
- **Torch-like APIs**: Familiar APIs for seamless transition from standard PyTorch to secure computation
- **Multiple Security Models**: Supports both Semi-Honest and Honest-Majority security assumptions
- **Flexible Party Configurations**: 2-party and 3-party computation setups
- **Multiple Secret Sharing Schemes**:
  - Additive Secret Sharing (2-party)
  - Replicated Secret Sharing (3-party)
- **Function Secret Sharing (FSS)** implementations with multiple variants:
- **Privacy-Preserving Neural Network Inference**: Support for secure model evaluation
- **Ring-based Computation**: All operations performed on finite rings for cryptographic security

## System Requirements

- **OS**: Linux is the primary supported platform; Windows works for CPU-only
  installs (and CUDA installs with matching toolchain) but is less tested.
- **Python**: 3.10 or higher (recommended: 3.12)
- **PyTorch**: >=2.5.0 (recommended: 2.7.1 or newer compatible release)
- **C/C++ compiler**: required because `torchcsprng` always builds a native
  extension. On Linux: `gcc`/`g++` (e.g. `sudo apt-get install build-essential`).
  On Windows: install
  [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  with the "Desktop development with C++" workload.
- **CUDA toolkit**: optional, only for GPU acceleration. Match the toolkit version
  to `torch.version.cuda`.

## Installation

NssMPClib bundles CUDA extensions and CUTLASS submodules. The included advice
script inspects your environment (Python, PyTorch, CUDA, nvcc, GPU, submodules)
and reports whether the machine is ready to install. It is read-only and never
installs anything itself; when something is missing, it names the required
package or version instead of trying to generate platform-specific commands.

### Step 1: Clone with submodules

```bash
git clone --recursive https://github.com/XidianNSS/NssMPClib.git
cd NssMPClib
```

If you cloned without `--recursive`, run `git submodule update --init --recursive`.

### Step 2: Check your environment

```bash
python3 scripts/installation_advice.py
```

If prerequisites (PyTorch, matching CUDA Toolkit / nvcc, submodules) are
missing, the script prints a `FAIL` item with the required version or condition.
Apply the fix that matches your OS/package manager and rerun the script until
the diagnosis passes or only reports intentional warnings.

### Step 3: Install NssMPClib

Once the check passes, the standard editable install is:

```bash
pip install -e . --no-build-isolation
```

Because `--no-build-isolation` reuses your environment instead of bootstrapping
a clean one, `setuptools` and `wheel` must already be installed there. The
advice script flags it explicitly if either is missing; install them with
`pip install --upgrade setuptools wheel` and rerun.

- **CUDA torch + matching nvcc + GPU visible**: `setup.py` auto-detects
  `CUDA_HOME` (by scanning `/usr/local/cuda-*` for the nvcc release matching
  `torch.version.cuda`) and `TORCH_CUDA_ARCH_LIST` (from visible GPUs), then
  builds the CUTLASS and CUDA `torchcsprng` extensions.
- **CPU-only torch**: `setup.py` skips the CUTLASS extension (since
  `torch.version.cuda` is unset) and `csprng/setup.py` skips its CUDA build
  (since `torch.cuda.is_available()` is False), so the same command above works
  as-is — no env vars needed.

The `NSSMPC_SKIP_CUTLASS=1 NSSMPC_SKIP_CSPRNG_CUDA=1` variant is only needed in
edge cases (CUDA torch installed but nvcc missing / no GPU / broken toolchain);
the advice script reports when those skip flags are already part of the selected
installation path.

### Step 4: Generate cryptographic parameters

```bash
python3 scripts/offline_parameter_generation.py
```

**Note**: Parameters are saved to `~/NssMPClib/data/` (32-bit in `data/32/`, 64-bit in `data/64/`).

## Quick Start: 2-Party Computation Example

**Party 0 - `party_0.py`**:

```python
from nssmpc import Party2PC, PartyRuntime, SEMI_HONEST, SecretTensor
import torch

party = Party2PC(0, SEMI_HONEST)
with PartyRuntime(party):
    party.online()
    x = torch.rand([10, 10])
    share_x = SecretTensor(tensor=x)
    result = share_x.recon().convert_to_real_field()
    print("Server result:", result)
```

**Party 1 - `party_1.py`**:

```python
from nssmpc import Party2PC, PartyRuntime, SEMI_HONEST, SecretTensor

client = Party2PC(1, SEMI_HONEST)
with PartyRuntime(client):
    client.online()
    share_x = SecretTensor(src_id=0)
    result = share_x.recon().convert_to_real_field()
    print("Client result:", result)
```

**Execution**:
```bash
# Terminal 1: Start server
python party_0.py

# Terminal 2: Start client (in separate terminal)
python party_1.py
```

## Running Built-in Examples

### 1. Arithmetic Secret Sharing (2-Party)
```bash
cd tests/primitives/secret_sharing/
# Terminal 1:
python -m unittest test_ass_p0.py
# Terminal 2:
python -m unittest test_ass_p1.py
```

### 2. Neural Network Inference (2-Party)
```bash
cd tests/application/neural_network/2pc/
# Terminal 1:
python neural_network_P0.py
# Terminal 2:
python neural_network_P1.py
```

### 3. Replicated Secret Sharing (3-Party)
```bash
cd tests/primitives/secret_sharing/
# Terminal 1: python -m unittest test_rss_p0.py
# Terminal 2: python -m unittest test_rss_p1.py  
# Terminal 3: python -m unittest test_rss_p2.py
```

## Configuration

Configure the library in `nssmpc/config/configs.json`:
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
├── nssmpc/                   # Main library source
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
   python3 scripts/offline_parameter_generation.py
   ```

2. **Port Already in Use**:
   Change base port in `configs.json` or kill existing processes.

3. **CUDA Errors**:
   Set `DEVICE: "cpu"` in config or check CUDA installation.

4. **Install-time CUDA / submodule errors** (e.g. `RuntimeError: The detected
   CUDA version (X.Y) mismatches ...`, or `fatal error: cutlass/...: No such
   file or directory`):
   Rerun the advice script. It will tell you whether the missing requirement is
   a matching CUDA Toolkit / nvcc version, a compatible PyTorch build, missing
   submodules, or an intentional CPU/skip-CUDA path:
   ```bash
   python3 scripts/installation_advice.py
   ```

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
