import os
import sys
from setuptools import setup, find_packages


def auto_fix_cuda_arch():
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError:
        print("Error: torch must be installed to build CUDA extensions.")
        sys.exit(1)

    env_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")

    if not env_arch or env_arch == "Common":
        print(
            f"Notice: TORCH_CUDA_ARCH_LIST is '{env_arch}'. Attempting auto-detection...")

        try:
            if not torch.cuda.is_available():
                detected_arch = "8.0 8.6"
                print("Warning: No GPU detected. Fallback to: " + detected_arch)
            else:
                arch_list = []
                for i in range(torch.cuda.device_count()):
                    cap = torch.cuda.get_device_capability(i)
                    arch = f"{cap[0]}.{cap[1]}"
                    if arch not in arch_list:
                        arch_list.append(arch)
                detected_arch = " ".join(arch_list)
                print(f"Auto-detected CUDA Architectures: {detected_arch}")
                
            os.environ["TORCH_CUDA_ARCH_LIST"] = detected_arch

        except Exception as e:
            print(
                f"Warning: Auto-detection failed ({e}). using default 8.0 8.6")
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0 8.6"
    else:
        print(f"Using existing TORCH_CUDA_ARCH_LIST: {env_arch}")

    return BuildExtension, CUDAExtension


BuildExtension, CUDAExtension = auto_fix_cuda_arch()

base_path = os.path.dirname(os.path.abspath(__file__))

cutlass_include_dir = os.path.join(base_path, 'cutlass', 'include')
cutlass_util_include_dir = os.path.join(
    base_path, 'cutlass', 'tools', 'util', 'include')

kernel_source_file = os.path.join(
    'nssmpc', 'infra', 'tensor', 'nss_cutlass_kernels.cu')

if not os.path.exists(kernel_source_file):
    print(f"Error: Source file not found at {kernel_source_file}")
    sys.exit(1)

nvcc_flags = [
    '-O3',
    '-std=c++17',                   
    '--expt-relaxed-constexpr',
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
]

cxx_flags = ['-O3', '-std=c++17']

ext_modules = [
    CUDAExtension(
        name='nssmpc.infra.tensor.cutlass_kernels',
        sources=[kernel_source_file],
        include_dirs=[
            cutlass_include_dir,
            cutlass_util_include_dir,
            os.path.dirname(kernel_source_file)
        ],
        extra_compile_args={
            'cxx': cxx_flags,
            'nvcc': nvcc_flags,
        },
    )
]

setup(
    name="NssMPClib",
    version="1.0.0b1",
    author="XDU NSS Lab",
    author_email="nss@xidian.edu.cn",
    description="A General-Purpose Secure Multi-Party Computation Library Based on PyTorch",
    url="https://gitcode.com/openHiTLS/NssMPClib",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        f'torchcsprng @ file://localhost/{base_path}/csprng',
        'torch>=2.3.0',
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Security :: Cryptography",
    ],
)