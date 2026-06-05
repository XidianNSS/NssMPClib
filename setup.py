import glob
import importlib.metadata
import os
import re
import shlex
import shutil
import site
import subprocess
import sys

from setuptools import find_packages, setup


YES_VALUES = ("1", "true", "yes", "on")
MIN_SETUPTOOLS_VERSION = (64,)
MIN_TORCH_VERSION = (2, 5, 0)


def _version_text(version):
    return ".".join(str(part) for part in version)


def _parse_numeric_version(value):
    if not value:
        return None
    match = re.match(r"^\s*(\d+(?:\.\d+)*)", value)
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _version_at_least(value, minimum):
    parsed = _parse_numeric_version(value)
    if not parsed:
        return False
    size = max(len(parsed), len(minimum))
    padded = parsed + (0,) * (size - len(parsed))
    padded_minimum = minimum + (0,) * (size - len(minimum))
    return padded >= padded_minimum


def _version_less_than(value, maximum):
    parsed = _parse_numeric_version(value)
    if not parsed:
        return False
    size = max(len(parsed), len(maximum))
    padded = parsed + (0,) * (size - len(parsed))
    padded_maximum = maximum + (0,) * (size - len(maximum))
    return padded < padded_maximum


def _env_enabled(name):
    return os.environ.get(name, "").strip().lower() in YES_VALUES


def _ensure_submodules():
    base = os.path.dirname(os.path.abspath(__file__))
    required = {
        "cutlass headers": os.path.join(base, "cutlass", "include", "cutlass", "cutlass.h"),
        "bundled torchcsprng": os.path.join(base, "csprng", "setup.py"),
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not os.path.exists(path)]
    if missing:
        sys.exit(
            "Error: required source trees are missing:\n  "
            + "\n  ".join(missing)
            + "\nRequired: initialize or restore the missing Git submodules.\n"
            "Run scripts/installation_advice.py to inspect the current environment."
        )


def _nvcc_release(cuda_home):
    if not cuda_home:
        return None
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.isfile(nvcc):
        return None
    try:
        out = subprocess.check_output([nvcc, "--version"], stderr=subprocess.STDOUT).decode("utf-8", "ignore")
    except (OSError, subprocess.CalledProcessError):
        return None
    m = re.search(r"release (\d+\.\d+)", out)
    return m.group(1) if m else None


def _dedupe_existing_paths(paths):
    result = []
    seen = set()

    for path in paths:
        if not path:
            continue

        path = os.path.abspath(path)
        if path in seen:
            continue

        if os.path.isdir(path):
            seen.add(path)
            result.append(path)

    return result


def _cuda_home_candidates():
    candidates = []

    for env_name in ("CUDA_HOME", "CUDA_PATH", "CONDA_PREFIX"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(value)

    path_nvcc = shutil.which("nvcc")
    if path_nvcc:
        candidates.append(os.path.dirname(os.path.dirname(path_nvcc)))

    candidates.append("/usr/local/cuda")
    candidates.extend(sorted(glob.glob("/usr/local/cuda-*"), reverse=True))

    return _dedupe_existing_paths(candidates)


def _python_site_dirs():
    dirs = []

    try:
        dirs.extend(site.getsitepackages())
    except Exception:
        pass

    try:
        user_site = site.getusersitepackages()
        if user_site:
            dirs.append(user_site)
    except Exception:
        pass

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        dirs.extend(glob.glob(os.path.join(conda_prefix, "lib", "python*", "site-packages")))

    return _dedupe_existing_paths(dirs)


def _cuda_include_dirs():
    candidates = []

    for root in _cuda_home_candidates():
        candidates.append(os.path.join(root, "include"))
        candidates.append(os.path.join(root, "targets", "x86_64-linux", "include"))

    for site_dir in _python_site_dirs():
        candidates.extend(glob.glob(os.path.join(site_dir, "nvidia", "*", "include")))

    return _dedupe_existing_paths(candidates)


def _cuda_library_dirs():
    candidates = []

    for root in _cuda_home_candidates():
        candidates.extend(
            [
                os.path.join(root, "lib64"),
                os.path.join(root, "lib"),
                os.path.join(root, "targets", "x86_64-linux", "lib"),
                os.path.join(root, "targets", "x86_64-linux", "lib", "stubs"),
            ]
        )

    for site_dir in _python_site_dirs():
        candidates.extend(glob.glob(os.path.join(site_dir, "nvidia", "*", "lib")))

    return _dedupe_existing_paths(candidates)


def _ensure_cuda_headers(include_dirs, torch_cuda):
    missing = []

    for header in ("cuda_runtime.h", "cublas_v2.h"):
        if not any(os.path.exists(os.path.join(include_dir, header)) for include_dir in include_dirs):
            missing.append(header)

    if not missing:
        return

    lines = [
        "Error: CUDA development headers are missing.",
        "Missing headers: " + ", ".join(missing),
        f"Required: CUDA development headers matching torch.version.cuda={torch_cuda}.",
        "Checked include directories:",
    ]

    if include_dirs:
        lines.extend(f"  {include_dir}" for include_dir in include_dirs)
    else:
        lines.append("  <none>")

    lines.append("Run scripts/installation_advice.py to inspect the current environment.")
    sys.exit("\n".join(lines))


def _cxx_compiler_command():
    env_cxx = os.environ.get("CXX")
    if env_cxx:
        return shlex.split(env_cxx)

    if sys.platform.startswith("win"):
        return ["cl"] if shutil.which("cl") else None

    for name in ("g++", "c++", "clang++"):
        if shutil.which(name):
            return [name]

    return None


def _compiler_version(command):
    for args in (command + ["-dumpfullversion", "-dumpversion"], command + ["--version"]):
        try:
            out = subprocess.check_output(args, stderr=subprocess.STDOUT).decode("utf-8", "ignore")
        except (OSError, subprocess.CalledProcessError):
            continue

        parsed = _parse_numeric_version(out)
        if parsed:
            return _version_text(parsed)

    return None


def _compiler_kind(command):
    name = os.path.basename(command[0]).lower()
    if "clang" in name:
        return "clang"
    return "gcc"


def _cuda_compiler_issue(torch_cuda, cpp_extension_module):
    if not torch_cuda or not sys.platform.startswith("linux"):
        return None

    if os.environ.get("TORCH_DONT_CHECK_COMPILER_ABI", "").upper() in (
        "ON",
        "1",
        "YES",
        "TRUE",
        "Y",
    ):
        return None

    command = _cxx_compiler_command()
    if not command:
        return "No C++ compiler command was found for CUDA extension builds."

    version = _compiler_version(command)
    if not version:
        return f"Could not determine C++ compiler version for {' '.join(command)}."

    kind = _compiler_kind(command)
    bounds_map = (
        getattr(cpp_extension_module, "CUDA_CLANG_VERSIONS", {})
        if kind == "clang"
        else getattr(cpp_extension_module, "CUDA_GCC_VERSIONS", {})
    )
    bounds = bounds_map.get(torch_cuda)
    if not bounds:
        return None

    min_version, max_exclusive_version = tuple(bounds[0]), tuple(bounds[1])
    if _version_at_least(version, min_version) and _version_less_than(version, max_exclusive_version):
        return None

    compiler_name = "clang++" if kind == "clang" else "g++"
    return (
        f"Detected {compiler_name}-compatible compiler {' '.join(command)} {version}, "
        f"but CUDA {torch_cuda} requires {compiler_name} "
        f">= {_version_text(min_version)}, < {_version_text(max_exclusive_version)} "
        "for PyTorch CUDA extension builds."
    )


def _ensure_cuda_compiler_compatible(torch_cuda, cpp_extension_module):
    issue = _cuda_compiler_issue(torch_cuda, cpp_extension_module)
    if issue:
        sys.exit(
            "Error: C++ compiler version is incompatible with this CUDA/PyTorch build.\n"
            f"Reason: {issue}\n"
            "Required: use a host C++ compiler version within PyTorch's CUDA compiler bounds, "
            "or use the intentional CPU/skip-CUDA install path.\n"
            "Run scripts/installation_advice.py to inspect the current environment."
        )


def _auto_set_cuda_home(torch_cuda):
    """Pick a CUDA root whose nvcc release exactly matches torch.version.cuda.

    PyTorch's cpp_extension compares torch.version.cuda against the system nvcc
    and may refuse to build on mismatch. Keep this in sync with
    scripts/installation_advice.py and csprng/setup.py.
    """
    if not torch_cuda:
        return True

    for candidate in _cuda_home_candidates():
        if _nvcc_release(candidate) == torch_cuda:
            os.environ["CUDA_HOME"] = candidate
            os.environ.setdefault("CUDA_PATH", candidate)
            print(
                f"Notice: auto-set CUDA_HOME={candidate} "
                f"(matches torch.version.cuda={torch_cuda})"
            )
            return True

    return False


def _set_torch_cuda_arch_list(torch):
    env_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env_arch and env_arch != "Common":
        print(f"Using existing TORCH_CUDA_ARCH_LIST: {env_arch}")
        return

    fallback = "8.0 8.6"
    try:
        if not torch.cuda.is_available():
            os.environ["TORCH_CUDA_ARCH_LIST"] = fallback
            print(f"Notice: no GPU detected at build time; defaulting TORCH_CUDA_ARCH_LIST={fallback}")
            return
        arch_list = []
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            arch = f"{cap[0]}.{cap[1]}"
            if arch not in arch_list:
                arch_list.append(arch)
        detected = " ".join(arch_list) or fallback
        os.environ["TORCH_CUDA_ARCH_LIST"] = detected
        print(f"Auto-detected TORCH_CUDA_ARCH_LIST={detected}")
    except Exception as e:
        os.environ["TORCH_CUDA_ARCH_LIST"] = fallback
        print(f"Warning: CUDA arch auto-detection failed ({e}); defaulted to {fallback}")


def _ensure_cpp_compiler():
    """torchcsprng always builds a native extension, so a C/C++ compiler is required."""
    import glob
    import platform
    import shutil

    if platform.system() == "Windows":
        if shutil.which("cl"):
            return
        # Look for an actual cl.exe inside common VS / Build Tools layouts —
        # directory existence alone isn't enough (installer leaves empty stubs).
        cl_patterns = [
            r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Host*\*\cl.exe",
            r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Host*\*\cl.exe",
            r"C:\BuildTools\VC\Tools\MSVC\*\bin\Host*\*\cl.exe",
        ]
        if any(glob.glob(p) for p in cl_patterns):
            sys.exit(
                "Error: Microsoft Visual C++ appears to be installed but 'cl' is not on PATH.\n"
                "Required: make the Visual C++ compiler reachable on PATH before building.\n"
                "Run scripts/installation_advice.py to inspect the current environment."
            )
        sys.exit(
            "Error: no C/C++ compiler was found, but torchcsprng's native extension "
            "must be compiled from source.\n"
            "Required on Windows: Microsoft Visual C++ 14.0 or newer, with 'cl' on PATH.\n"
            "Run scripts/installation_advice.py to inspect the current environment."
        )

    for name in ("c++", "g++", "clang++", "cc", "gcc", "clang"):
        if shutil.which(name):
            return
    sys.exit(
        "Error: no C/C++ compiler was found, but torchcsprng's native extension "
        "must be compiled from source.\n"
        "Required on Linux: a system C/C++ compiler such as gcc/g++ or clang.\n"
        "Run scripts/installation_advice.py to inspect the current environment."
    )


def _ensure_python_build_deps():
    try:
        setuptools_version = importlib.metadata.version("setuptools")
    except importlib.metadata.PackageNotFoundError:
        setuptools_version = None

    if not setuptools_version or not _version_at_least(setuptools_version, MIN_SETUPTOOLS_VERSION):
        current = setuptools_version or "missing"
        sys.exit(
            "Error: setuptools is required to build NssMPClib with "
            f"--no-build-isolation, and must be >= {_version_text(MIN_SETUPTOOLS_VERSION)}.\n"
            f"Current setuptools: {current}\n"
            "Run scripts/installation_advice.py to inspect the current environment."
        )

    try:
        import wheel  # noqa: F401
    except ImportError:
        sys.exit(
            "Error: the 'wheel' package is required to build NssMPClib with "
            "--no-build-isolation, but it is not installed in this environment.\n"
            "Run scripts/installation_advice.py to inspect the current environment."
        )


def _ensure_torch_version(torch):
    version = getattr(torch, "__version__", None)
    if _version_at_least(version, MIN_TORCH_VERSION):
        return

    sys.exit(
        "Error: PyTorch is too old for NssMPClib.\n"
        f"Required: torch >= {_version_text(MIN_TORCH_VERSION)}.\n"
        f"Current torch: {version or '<unknown>'}.\n"
        "Run scripts/installation_advice.py to inspect the current environment."
    )


_ensure_submodules()
_ensure_cpp_compiler()
_ensure_python_build_deps()

try:
    import torch
except ImportError:
    sys.exit(
        "Error: torch must be installed before building NssMPClib.\n"
        "Required: torch >= 2.5.0, with a CPU or CUDA variant chosen intentionally.\n"
        "Run scripts/installation_advice.py to inspect the current environment."
    )

_ensure_torch_version(torch)

from torch.utils import cpp_extension  # noqa: E402
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402

if os.environ.get("CUDA_HOME"):
    cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]

base_path = os.path.dirname(os.path.abspath(__file__))

cutlass_include_dir = os.path.join(base_path, "cutlass", "include")
cutlass_util_include_dir = os.path.join(base_path, "cutlass", "tools", "util", "include")
kernel_source_file = os.path.join("nssmpc", "infra", "utils", "nss_cutlass_kernels.cu")

SKIP_CUTLASS = _env_enabled("NSSMPC_SKIP_CUTLASS")
SKIP_CSPRNG_CUDA = _env_enabled("NSSMPC_SKIP_CSPRNG_CUDA")
FORCE_CUDA = os.environ.get("FORCE_CUDA", "0") == "1"

if torch.version.cuda and (SKIP_CUTLASS or SKIP_CSPRNG_CUDA) and not (
    SKIP_CUTLASS and SKIP_CSPRNG_CUDA
):
    sys.exit(
        "Error: CUDA extension skip flags are incomplete.\n"
        "Required for an intentional CPU/skip-CUDA NssMPClib install: set both "
        "NSSMPC_SKIP_CUTLASS=1 and NSSMPC_SKIP_CSPRNG_CUDA=1.\n"
        "Alternatively, unset both flags and satisfy the CUDA build requirements.\n"
        "Run scripts/installation_advice.py to inspect the current environment."
    )

ext_modules = []
cmdclass = {}

if SKIP_CUTLASS:
    print(
        "Notice: NSSMPC_SKIP_CUTLASS is set; skipping CUTLASS extension. "
        "The runtime will fall back to the pure-PyTorch matmul path "
        "(see nssmpc/infra/utils/cuda.py)."
    )
elif not torch.version.cuda:
    print(
        "Notice: torch is CPU-only; skipping CUTLASS extension build. "
        "Install a CUDA-enabled torch and reinstall to enable the fast matmul path."
    )
elif not torch.cuda.is_available() and not FORCE_CUDA:
    sys.exit(
        "Error: torch was built with CUDA, but torch.cuda.is_available() is false.\n"
        "NssMPClib will not build CUDA extensions from this ambiguous state.\n"
        "Required for the CUDA path: GPU/runtime visible to PyTorch, matching nvcc, "
        "and CUDA development headers.\n"
        "For an intentional CPU/skip-CUDA install, set NSSMPC_SKIP_CUTLASS=1 "
        "and NSSMPC_SKIP_CSPRNG_CUDA=1.\n"
        "Run scripts/installation_advice.py to inspect the current environment."
    )
elif not os.path.exists(kernel_source_file):
    print(f"Warning: kernel source {kernel_source_file} not found; skipping CUTLASS extension.")
else:
    if not _auto_set_cuda_home(torch.version.cuda):
        sys.exit(
            "Error: CUDA PyTorch is installed, but no matching CUDA Toolkit / nvcc was found.\n"
            f"Required: CUDA Toolkit / nvcc {torch.version.cuda}, matching torch.version.cuda.\n"
            "Run scripts/installation_advice.py to inspect the current environment."
        )

    if os.environ.get("CUDA_HOME"):
        cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]

    cuda_include_dirs = _cuda_include_dirs()
    cuda_library_dirs = _cuda_library_dirs()
    _ensure_cuda_headers(cuda_include_dirs, torch.version.cuda)
    _ensure_cuda_compiler_compatible(torch.version.cuda, cpp_extension)
    _set_torch_cuda_arch_list(torch)

    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
    ]
    cxx_flags = ["-O3", "-std=c++17"]
    ext_modules = [
        CUDAExtension(
            name="nssmpc.infra.utils.cutlass_kernels",
            sources=[kernel_source_file],
            include_dirs=[
                cutlass_include_dir,
                cutlass_util_include_dir,
                os.path.dirname(kernel_source_file),
            ]
            + cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

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
    cmdclass=cmdclass,
    install_requires=[
        f"torchcsprng @ file://{base_path}/csprng",
        "torch>=2.5.0",
    ],
    python_requires=">=3.10",
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
