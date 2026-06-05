#!/usr/bin/env python3
"""Diagnose whether the current machine is ready to install NssMPClib.

This script is intentionally read-only. It checks Python, PyTorch, CUDA,
nvcc, GPU architecture, submodule state, Python build dependencies, and a
C/C++ compiler.

It does NOT print installation commands. Instead, it reports PASS/WARN/FAIL
with the reason and expected condition, so users can decide how to fix their
own environment.
"""

from __future__ import annotations

import glob
import importlib.metadata
import os
import platform
import re
import shlex
import shutil
import site
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


YES_VALUES = {"1", "true", "yes", "on"}

MIN_PYTHON_VERSION = (3, 10)
RECOMMENDED_PYTHON_VERSION = "3.12"
MIN_TORCH_VERSION = (2, 5, 0)
RECOMMENDED_TORCH_VERSION = "2.7.1"
MIN_SETUPTOOLS_VERSION = (64,)

# CUDA versions for which PyTorch commonly publishes cu* wheel indexes.
# This is kept for diagnosis only; the script does not generate install commands.
PYTORCH_CUDA_INDEXES = ("11.8", "12.1", "12.4", "12.6", "12.8")

# Minimum CUDA compute capability supported by modern PyTorch CUDA wheels.
# Cards below this, for example Kepler sm_3.x, cannot run recent cu* wheels.
MIN_PYTORCH_COMPUTE_CAP = (5, 0)


@dataclass(frozen=True)
class NvccInfo:
    path: str
    release: str | None


@dataclass(frozen=True)
class TorchInfo:
    installed: bool
    version: str | None = None
    cuda_version: str | None = None
    cuda_available: bool = False
    devices: tuple[tuple[int, str, str], ...] = ()
    error: str | None = None


def run_cmd(args: list[str], timeout: int = 5) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return proc.returncode, proc.stdout.strip()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in YES_VALUES


def version_text(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def _parse_numeric_version(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None

    match = re.match(r"^\s*(\d+(?:\.\d+)*)", value)
    if not match:
        return None

    return tuple(int(part) for part in match.group(1).split("."))


def version_at_least(value: str | None, minimum: tuple[int, ...]) -> bool:
    parsed = _parse_numeric_version(value)
    if not parsed:
        return False

    size = max(len(parsed), len(minimum))
    padded = parsed + (0,) * (size - len(parsed))
    padded_minimum = minimum + (0,) * (size - len(minimum))
    return padded >= padded_minimum


def version_less_than(value: str | None, maximum: tuple[int, ...]) -> bool:
    parsed = _parse_numeric_version(value)
    if not parsed:
        return False

    size = max(len(parsed), len(maximum))
    padded = parsed + (0,) * (size - len(parsed))
    padded_maximum = maximum + (0,) * (size - len(maximum))
    return padded < padded_maximum


def cuda_label(cuda_version: str) -> str:
    return "cu" + cuda_version.replace(".", "")


def known_cuda_labels() -> str:
    return ", ".join(cuda_label(version) for version in PYTORCH_CUDA_INDEXES)


def detect_torch() -> TorchInfo:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return TorchInfo(installed=False, error=str(exc))

    devices: list[tuple[int, str, str]] = []
    cuda_available = False

    try:
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            for idx in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(idx)
                devices.append((idx, torch.cuda.get_device_name(idx), f"{major}.{minor}"))
    except Exception as exc:
        return TorchInfo(
            installed=True,
            version=getattr(torch, "__version__", None),
            cuda_version=getattr(torch.version, "cuda", None),
            cuda_available=False,
            devices=tuple(devices),
            error=f"torch CUDA probe failed: {exc}",
        )

    return TorchInfo(
        installed=True,
        version=getattr(torch, "__version__", None),
        cuda_version=getattr(torch.version, "cuda", None),
        cuda_available=cuda_available,
        devices=tuple(devices),
    )


def nvcc_release(nvcc_path: str) -> str | None:
    code, out = run_cmd([nvcc_path, "--version"])
    if code != 0:
        return None

    match = re.search(r"release\s+(\d+\.\d+)", out)
    return match.group(1) if match else None


def unique_paths(paths: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        result.append(path)

    return result


def unique_existing_dirs(paths: Iterable[str]) -> list[str]:
    result: list[str] = []

    for path in unique_paths(paths):
        if Path(path).is_dir():
            result.append(path)

    return result


def cuda_home_candidates() -> list[str]:
    candidates: list[str] = []

    for env_name in ("CUDA_HOME", "CUDA_PATH", "CONDA_PREFIX"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(value)

    path_nvcc = shutil.which("nvcc")
    if path_nvcc:
        candidates.append(str(Path(path_nvcc).parent.parent))

    candidates.append("/usr/local/cuda")
    candidates.extend(sorted(glob.glob("/usr/local/cuda-*"), reverse=True))

    return unique_paths(str(Path(candidate).absolute()) for candidate in candidates if candidate)


def detect_nvccs() -> list[NvccInfo]:
    candidates: list[str] = []

    for cuda_home in cuda_home_candidates():
        candidates.append(str(Path(cuda_home) / "bin" / "nvcc"))

    infos: list[NvccInfo] = []
    for path in unique_paths(candidates):
        if Path(path).is_file():
            infos.append(NvccInfo(path=path, release=nvcc_release(path)))

    return infos


def python_site_dirs() -> list[str]:
    dirs: list[str] = []

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
        dirs.extend(glob.glob(str(Path(conda_prefix) / "lib" / "python*" / "site-packages")))

    return unique_existing_dirs(dirs)


def cuda_include_dirs() -> list[str]:
    candidates: list[str] = []

    for root in cuda_home_candidates():
        candidates.append(str(Path(root) / "include"))
        candidates.append(str(Path(root) / "targets" / "x86_64-linux" / "include"))

    for site_dir in python_site_dirs():
        candidates.extend(glob.glob(str(Path(site_dir) / "nvidia" / "*" / "include")))

    return unique_existing_dirs(candidates)


def missing_cuda_headers() -> tuple[list[str], list[str]]:
    include_dirs = cuda_include_dirs()
    missing: list[str] = []

    for header in ("cuda_runtime.h", "cublas_v2.h"):
        if not any((Path(include_dir) / header).exists() for include_dir in include_dirs):
            missing.append(header)

    return missing, include_dirs


def detect_nvidia_smi() -> tuple[bool, str | None]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False, None

    code, out = run_cmd([smi, "-L"])
    if code != 0 or not out:
        return True, None

    return True, out


def _parse_version(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None

    m = re.match(r"^\s*(\d+)\.(\d+)", value)
    return (int(m.group(1)), int(m.group(2))) if m else None


def nvidia_smi_driver_cuda() -> str | None:
    """Return max CUDA version supported by the NVIDIA driver, parsed from nvidia-smi."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None

    code, out = run_cmd([smi])
    if code != 0:
        return None

    m = re.search(r"CUDA Version:\s*([\d.]+)", out)
    return m.group(1) if m else None


def nvidia_smi_compute_caps() -> list[str]:
    """Return per-GPU compute capability strings from nvidia-smi."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []

    code, out = run_cmd([smi, "--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
    if code != 0:
        return []

    return [line.strip() for line in out.splitlines() if line.strip()]


def pytorch_cu_index_for(cuda_version: str | None) -> str | None:
    """Return the highest known PyTorch CUDA index <= cuda_version.

    Used only for diagnosis text. This script does not generate pip commands.
    """
    target = _parse_version(cuda_version)
    if not target:
        return None

    best: tuple[int, int] | None = None
    best_str: str | None = None

    for version in PYTORCH_CUDA_INDEXES:
        parsed = _parse_version(version)
        if parsed and parsed <= target and (best is None or parsed > best):
            best = parsed
            best_str = version

    return best_str


def all_compute_caps_too_old(caps: Iterable[str]) -> bool:
    """True iff at least one cap was reported and the max is below MIN_PYTORCH_COMPUTE_CAP."""
    parsed = [p for p in (_parse_version(cap) for cap in caps) if p]
    if not parsed:
        return False

    return max(parsed) < MIN_PYTORCH_COMPUTE_CAP


def matching_nvcc(torch_cuda: str | None, nvccs: list[NvccInfo]) -> NvccInfo | None:
    if not torch_cuda:
        return None

    for info in nvccs:
        if info.release == torch_cuda:
            return info

    return None


def major_matching_nvcc(torch_cuda: str | None, nvccs: list[NvccInfo]) -> NvccInfo | None:
    if not torch_cuda:
        return None

    major = torch_cuda.split(".", 1)[0]

    for info in nvccs:
        if info.release and info.release.split(".", 1)[0] == major:
            return info

    return None


def preferred_nvcc(nvccs: list[NvccInfo]) -> NvccInfo | None:
    versioned = [info for info in nvccs if info.release]
    if not versioned:
        return nvccs[0] if nvccs else None

    def key(info: NvccInfo) -> tuple[int, int]:
        major, _, minor = (info.release or "0.0").partition(".")
        return int(major or 0), int(minor or 0)

    return max(versioned, key=key)


def submodule_status(root: Path) -> list[str]:
    missing: list[str] = []

    if not (root / "cutlass" / "include" / "cutlass" / "cutlass.h").exists():
        missing.append("cutlass")

    if not (root / "csprng" / "setup.py").exists():
        missing.append("csprng")

    return missing


def detect_missing_build_deps() -> list[str]:
    """Return missing or too-old Python build dependencies."""
    issues: list[str] = []

    setuptools_version = package_version("setuptools")
    if setuptools_version is None:
        issues.append(f"setuptools missing (required >= {version_text(MIN_SETUPTOOLS_VERSION)})")
    elif not version_at_least(setuptools_version, MIN_SETUPTOOLS_VERSION):
        issues.append(
            f"setuptools {setuptools_version} "
            f"(required >= {version_text(MIN_SETUPTOOLS_VERSION)})"
        )

    wheel_version = package_version("wheel")
    if wheel_version is None:
        issues.append("wheel missing (required: installed)")

    return issues


def package_version(name: str) -> str | None:
    try:
        mod = __import__(name)
    except ImportError:
        return None

    version = getattr(mod, "__version__", None)
    if version:
        return str(version)

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def detect_cpp_compiler() -> tuple[bool, str | None]:
    """Best-effort detection of a C/C++ compiler usable for extension builds."""
    if platform.system() == "Windows":
        if shutil.which("cl"):
            return True, None

        cl_patterns = [
            r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Host*\*\cl.exe",
            r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Host*\*\cl.exe",
            r"C:\BuildTools\VC\Tools\MSVC\*\bin\Host*\*\cl.exe",
        ]

        if any(glob.glob(pattern) for pattern in cl_patterns):
            return False, (
                "Microsoft Visual C++ appears to be installed, but 'cl' is not on PATH. "
                "Open a Visual Studio Native Tools shell or initialize the compiler "
                "environment before retrying."
            )

        return False, (
            "Building native extensions requires Microsoft Visual C++ 14.0 or newer."
        )

    for name in ("c++", "g++", "clang++", "cc", "gcc", "clang"):
        if shutil.which(name):
            return True, None

    return False, (
        "Building native extensions requires a system C/C++ compiler such as gcc/g++ or clang."
    )


def cxx_compiler_command() -> list[str] | None:
    env_cxx = os.environ.get("CXX")
    if env_cxx:
        return shlex.split(env_cxx)

    if platform.system() == "Windows":
        return ["cl"] if shutil.which("cl") else None

    for name in ("g++", "c++", "clang++"):
        if shutil.which(name):
            return [name]

    return None


def compiler_version(command: list[str]) -> str | None:
    for args in (command + ["-dumpfullversion", "-dumpversion"], command + ["--version"]):
        code, out = run_cmd(args)
        if code != 0:
            continue

        parsed = _parse_numeric_version(out)
        if parsed:
            return version_text(parsed)

    return None


def compiler_kind(command: list[str]) -> str:
    name = Path(command[0]).name.lower()
    if "clang" in name:
        return "clang"
    return "gcc"


def cuda_compiler_bounds(torch_cuda: str, kind: str) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    try:
        from torch.utils import cpp_extension  # type: ignore
    except Exception:
        return None

    bounds_map = (
        getattr(cpp_extension, "CUDA_CLANG_VERSIONS", {})
        if kind == "clang"
        else getattr(cpp_extension, "CUDA_GCC_VERSIONS", {})
    )
    bounds = bounds_map.get(torch_cuda)
    if not bounds:
        return None

    return tuple(bounds[0]), tuple(bounds[1])


def detect_cuda_compiler_issue(torch_cuda: str | None) -> str | None:
    if not torch_cuda or platform.system() != "Linux":
        return None

    if os.environ.get("TORCH_DONT_CHECK_COMPILER_ABI", "").upper() in {
        "ON",
        "1",
        "YES",
        "TRUE",
        "Y",
    }:
        return None

    command = cxx_compiler_command()
    if not command:
        return "No C++ compiler command was found for CUDA extension builds."

    version = compiler_version(command)
    if not version:
        return f"Could not determine C++ compiler version for {' '.join(command)}."

    kind = compiler_kind(command)
    bounds = cuda_compiler_bounds(torch_cuda, kind)
    if not bounds:
        return None

    min_version, max_exclusive_version = bounds
    if not version_at_least(version, min_version) or not version_less_than(version, max_exclusive_version):
        compiler_name = "clang++" if kind == "clang" else "g++"
        return (
            f"Detected {compiler_name}-compatible compiler {' '.join(command)} {version}, "
            f"but CUDA {torch_cuda} requires {compiler_name} "
            f">= {version_text(min_version)}, < {version_text(max_exclusive_version)} "
            "for PyTorch CUDA extension builds."
        )

    return None


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def print_status(status: str, message: str) -> None:
    print(f"{status}: {message}")


def print_kv(key: str, value: str | None) -> None:
    print(f"{key}: {value if value else '<unknown>'}")


def print_required_versions(torch_info: TorchInfo, nvccs: list[NvccInfo]) -> None:
    print_section("Required versions")
    print(
        f"Python: >= {version_text(MIN_PYTHON_VERSION)} "
        f"(recommended {RECOMMENDED_PYTHON_VERSION})"
    )
    print(
        f"PyTorch: >= {version_text(MIN_TORCH_VERSION)} "
        f"(recommended {RECOMMENDED_TORCH_VERSION} or newer compatible release)"
    )
    print(f"Build backend: setuptools >= {version_text(MIN_SETUPTOOLS_VERSION)}, wheel installed")
    print("Native compiler: gcc/g++ or clang on Linux; Microsoft Visual C++ 14.0+ on Windows")

    if torch_info.cuda_version:
        print(
            "CUDA extension path: CUDA Toolkit / nvcc and development headers "
            f"must match torch.version.cuda ({torch_info.cuda_version})."
        )
        driver_cuda = nvidia_smi_driver_cuda()
        if driver_cuda:
            print(
                "NVIDIA driver: must support at least CUDA "
                f"{torch_info.cuda_version}; detected driver max is {driver_cuda}."
            )
        return

    has_smi, _ = detect_nvidia_smi()
    if has_smi:
        driver_cuda = nvidia_smi_driver_cuda()
        picked = pytorch_cu_index_for(driver_cuda)
        print("CPU path: CPU-only PyTorch is valid; CUDA Toolkit / nvcc is not required.")
        print(
            "GPU path: use a CUDA-enabled PyTorch build and a CUDA Toolkit / nvcc "
            "with the same CUDA X.Y version."
        )
        if driver_cuda:
            print(f"NVIDIA driver max CUDA: {driver_cuda}")
        if picked:
            print(f"Highest known PyTorch CUDA family fitting this driver: {cuda_label(picked)}")
        print(f"Known PyTorch CUDA families in this checker: {known_cuda_labels()}")
        return

    print("CPU path: CPU-only PyTorch is valid; CUDA Toolkit / nvcc is not required.")


def diagnose(root: Path, torch_info: TorchInfo, nvccs: list[NvccInfo]) -> int:
    """Diagnose installation readiness.

    Return codes:
      0: PASS or acceptable WARN state
      1: FAIL state requiring user action
    """
    missing_submodules = submodule_status(root)
    missing_build_deps = detect_missing_build_deps()
    skip_cutlass = env_enabled("NSSMPC_SKIP_CUTLASS")
    skip_csprng_cuda = env_enabled("NSSMPC_SKIP_CSPRNG_CUDA")
    force_cuda = env_enabled("FORCE_CUDA")

    print_section("Diagnosis")

    if sys.version_info < MIN_PYTHON_VERSION:
        print_status("FAIL", "Python is too old for NssMPClib.")
        print_kv("current Python", platform.python_version())
        print(
            "Required: Python "
            f"{version_text(MIN_PYTHON_VERSION)} or newer "
            f"(recommended {RECOMMENDED_PYTHON_VERSION})."
        )
        return 1

    if missing_submodules:
        print_status("FAIL", "Git submodules are missing.")
        print("Missing: " + ", ".join(missing_submodules))
        print("Reason: required source trees are not present in the repository.")
        print(
            "Required: the cutlass source tree must contain "
            "cutlass/include/cutlass/cutlass.h, and the csprng source tree "
            "must contain csprng/setup.py."
        )
        print("Expected: initialize or restore the missing submodules, then rerun this check.")
        return 1

    if missing_build_deps:
        print_status("FAIL", "Python build dependencies are missing.")
        print("Missing/outdated: " + ", ".join(missing_build_deps))
        print("Reason: editable builds with --no-build-isolation need these packages.")
        print(
            "Required: setuptools "
            f">= {version_text(MIN_SETUPTOOLS_VERSION)} and wheel installed "
            "in the same Python environment that will run pip."
        )
        print("Expected: make the required Python build packages available, then rerun this check.")
        return 1

    has_compiler, compiler_advice = detect_cpp_compiler()
    if not has_compiler:
        print_status("FAIL", "No usable C/C++ compiler was detected.")
        print("Reason: native extensions must be compiled from source.")
        if compiler_advice:
            print("Details: " + compiler_advice)
        print(
            "Required: gcc/g++ or clang on Linux; Microsoft Visual C++ 14.0+ "
            "with 'cl' on PATH on Windows."
        )
        print("Expected: make a C/C++ compiler available on PATH, then rerun this check.")
        return 1

    if not torch_info.installed:
        print_status("FAIL", "PyTorch is not installed.")
        if torch_info.error:
            print("Import error: " + torch_info.error)
        print("Reason: CUDA availability and extension build compatibility cannot be evaluated.")
        print(
            "Required: PyTorch "
            f">= {version_text(MIN_TORCH_VERSION)} "
            f"(recommended {RECOMMENDED_TORCH_VERSION} or newer compatible release)."
        )
        preferred = preferred_nvcc(nvccs)
        driver_cuda = nvidia_smi_driver_cuda()
        target_cuda = preferred.release if preferred and preferred.release else driver_cuda
        picked = pytorch_cu_index_for(target_cuda)
        if preferred and preferred.release and picked:
            print(
                "For GPU extension builds on this machine, prefer a CUDA PyTorch "
                f"build with torch.version.cuda={picked} because nvcc {preferred.release} "
                "was detected."
            )
        elif driver_cuda and picked:
            print(
                "For GPU use on this machine, choose a CUDA PyTorch build no newer "
                f"than driver-supported CUDA {driver_cuda}; the highest known family "
                f"here is {cuda_label(picked)}."
            )
        else:
            print("For CPU-only use, install a CPU-only PyTorch build; CUDA Toolkit / nvcc is not required.")
        print("Expected: provide a suitable PyTorch installation, then rerun this check.")
        return 1

    if torch_info.error:
        print_status("FAIL", "PyTorch was imported, but CUDA probing failed.")
        print("Details: " + torch_info.error)
        print("Reason: the PyTorch/CUDA runtime is not healthy enough to evaluate safely.")
        print("Expected: fix the PyTorch CUDA runtime error, then rerun this check.")
        return 1

    if not version_at_least(torch_info.version, MIN_TORCH_VERSION):
        print_status("FAIL", "PyTorch is too old for NssMPClib.")
        print_kv("current torch", torch_info.version)
        print(
            "Required: PyTorch "
            f"{version_text(MIN_TORCH_VERSION)} or newer "
            f"(recommended {RECOMMENDED_TORCH_VERSION} or newer compatible release)."
        )
        print(
            "Expected: keep the CPU/CUDA variant intentional; if using CUDA, "
            "torch.version.cuda must match the CUDA Toolkit / nvcc release."
        )
        return 1

    if skip_cutlass or skip_csprng_cuda:
        if torch_info.cuda_version and not (skip_cutlass and skip_csprng_cuda):
            print_status("FAIL", "CUDA extension skip flags are incomplete.")
            if skip_cutlass:
                print("  NSSMPC_SKIP_CUTLASS: enabled")
            else:
                print("  NSSMPC_SKIP_CUTLASS: not enabled")
            if skip_csprng_cuda:
                print("  NSSMPC_SKIP_CSPRNG_CUDA: enabled")
            else:
                print("  NSSMPC_SKIP_CSPRNG_CUDA: not enabled")
            print(
                "Reason: with a CUDA-enabled PyTorch build, the top-level CUTLASS "
                "extension and bundled torchcsprng CUDA extension are controlled "
                "by separate flags."
            )
            print(
                "Required for an intentional CPU/skip-CUDA path: enable both "
                "NSSMPC_SKIP_CUTLASS and NSSMPC_SKIP_CSPRNG_CUDA, or unset both "
                "and satisfy the CUDA build requirements."
            )
            return 1

        print_status("WARN", "CUDA extension skip flags are enabled.")
        if skip_cutlass:
            print("  NSSMPC_SKIP_CUTLASS: enabled")
        if skip_csprng_cuda:
            print("  NSSMPC_SKIP_CSPRNG_CUDA: enabled")
        print("Result: installation may proceed, but some CUDA extensions may be skipped.")
        print(
            "Required for this path: PyTorch "
            f">= {version_text(MIN_TORCH_VERSION)}, setuptools "
            f">= {version_text(MIN_SETUPTOOLS_VERSION)}, wheel, and a native C/C++ compiler."
        )
        return 0

    if torch_info.cuda_version and torch_info.cuda_available:
        exact = matching_nvcc(torch_info.cuda_version, nvccs)
        if exact:
            missing_headers, include_dirs = missing_cuda_headers()
            if missing_headers:
                print_status("FAIL", "CUDA development headers are missing.")
                print("Missing headers: " + ", ".join(missing_headers))
                print("Reason: native CUDA extensions need CUDA headers in addition to nvcc.")
                print(
                    "Required: CUDA development headers matching "
                    f"torch.version.cuda ({torch_info.cuda_version})."
                )
                print("Checked include directories:")
                if include_dirs:
                    for include_dir in include_dirs:
                        print(f"  {include_dir}")
                else:
                    print("  <none>")
                return 1

            compiler_issue = detect_cuda_compiler_issue(torch_info.cuda_version)
            if compiler_issue:
                print_status("FAIL", "C++ compiler version is incompatible with this CUDA/PyTorch build.")
                print("Reason: " + compiler_issue)
                print(
                    "Required: use a host C++ compiler version within PyTorch's "
                    f"CUDA {torch_info.cuda_version} bounds, or intentionally use the CPU/skip-CUDA path."
                )
                return 1

            print_status("PASS", "CUDA PyTorch and matching nvcc were detected.")
            print_kv("torch.version.cuda", torch_info.cuda_version)
            print_kv("nvcc path", exact.path)
            print_kv("nvcc release", exact.release)
            print("Result: environment is ready for CUDA extension builds.")
            return 0

        major_match = major_matching_nvcc(torch_info.cuda_version, nvccs)

        print_status("FAIL", "CUDA PyTorch is installed, but matching nvcc was not found.")
        print_kv("torch.version.cuda", torch_info.cuda_version)

        if major_match:
            print_kv("closest nvcc path", major_match.path)
            print_kv("closest nvcc release", major_match.release)
            print("Reason: an nvcc with the same major CUDA version exists, but the minor version differs.")
        elif nvccs:
            print("Detected nvcc versions:")
            for info in nvccs:
                print(f"  {info.path} release {info.release or 'unknown'}")
            print("Reason: none of the detected nvcc versions exactly match torch.version.cuda.")
        else:
            print("Detected nvcc: none")
            print("Reason: CUDA Toolkit compiler is missing, or nvcc is not visible through PATH/CUDA_HOME.")

        print(
            "Expected: provide an nvcc release matching torch.version.cuda, "
            "or use a PyTorch build that matches the available nvcc."
        )
        print(
            "Required CUDA version: "
            f"CUDA Toolkit / nvcc {torch_info.cuda_version} for this PyTorch build."
        )
        preferred = preferred_nvcc(nvccs)
        if preferred and preferred.release:
            picked = pytorch_cu_index_for(preferred.release)
            if picked:
                print(
                    "Alternative version target: use a CUDA PyTorch build with "
                    f"torch.version.cuda={picked} to match detected nvcc {preferred.release}."
                )
        print(
            "Note: PyTorch CUDA wheels can provide CUDA runtime libraries, "
            "but native CUDA extension builds still require an nvcc compiler."
        )
        return 1

    if torch_info.cuda_version and not torch_info.cuda_available:
        if force_cuda:
            exact = matching_nvcc(torch_info.cuda_version, nvccs)
            if not exact:
                print_status("FAIL", "FORCE_CUDA is enabled, but matching nvcc was not found.")
                print_kv("torch.version.cuda", torch_info.cuda_version)
                if nvccs:
                    print("Detected nvcc versions:")
                    for info in nvccs:
                        print(f"  {info.path} release {info.release or 'unknown'}")
                else:
                    print("Detected nvcc: none")
                print(
                    "Required: CUDA Toolkit / nvcc "
                    f"{torch_info.cuda_version}, matching torch.version.cuda."
                )
                return 1

            missing_headers, include_dirs = missing_cuda_headers()
            if missing_headers:
                print_status("FAIL", "FORCE_CUDA is enabled, but CUDA development headers are missing.")
                print("Missing headers: " + ", ".join(missing_headers))
                print(
                    "Required: CUDA development headers matching "
                    f"torch.version.cuda ({torch_info.cuda_version})."
                )
                print("Checked include directories:")
                if include_dirs:
                    for include_dir in include_dirs:
                        print(f"  {include_dir}")
                else:
                    print("  <none>")
                return 1

            compiler_issue = detect_cuda_compiler_issue(torch_info.cuda_version)
            if compiler_issue:
                print_status("FAIL", "FORCE_CUDA is enabled, but the C++ compiler is incompatible.")
                print("Reason: " + compiler_issue)
                print(
                    "Required: use a host C++ compiler version within PyTorch's "
                    f"CUDA {torch_info.cuda_version} bounds."
                )
                return 1

            print_status("WARN", "FORCE_CUDA is enabled while no GPU is visible to PyTorch.")
            print_kv("torch.version.cuda", torch_info.cuda_version)
            print_kv("nvcc path", exact.path)
            print_kv("nvcc release", exact.release)
            print(
                "Result: CUDA extension builds may proceed for an offline/headless build. "
                "Set TORCH_CUDA_ARCH_LIST explicitly if the default architectures are not appropriate."
            )
            return 0

        print_status("FAIL", "PyTorch was built with CUDA, but CUDA is not available at runtime.")
        print_kv("torch.version.cuda", torch_info.cuda_version)
        print("Reason: the NVIDIA driver, GPU visibility, container passthrough, or CUDA runtime may be misconfigured.")
        print(
            "Required for GPU path: NVIDIA driver/runtime visible to PyTorch, "
            f"plus CUDA Toolkit / nvcc {torch_info.cuda_version} for native CUDA extension builds."
        )
        driver_cuda = nvidia_smi_driver_cuda()
        if driver_cuda:
            print(f"Detected driver-supported CUDA max: {driver_cuda}")
        print(
            "Expected: make CUDA visible to PyTorch, or intentionally use a "
            f"CPU-only PyTorch build >= {version_text(MIN_TORCH_VERSION)}."
        )
        return 1

    has_nvidia_smi, smi_output = detect_nvidia_smi()
    if has_nvidia_smi:
        print_status("WARN", "PyTorch is CPU-only, but an NVIDIA driver/GPU was detected.")
        if smi_output:
            print("Detected GPUs:")
            for line in smi_output.splitlines():
                print(f"  {line}")

        compute_caps = nvidia_smi_compute_caps()
        if all_compute_caps_too_old(compute_caps):
            cap_str = ", ".join(compute_caps) if compute_caps else "<unknown>"
            min_cap = f"sm_{MIN_PYTORCH_COMPUTE_CAP[0]}{MIN_PYTORCH_COMPUTE_CAP[1]}"
            print(f"Reason: reported GPU compute capability ({cap_str}) is below {min_cap}.")
            print("Result: CPU-only installation is likely the appropriate path.")
            return 0

        driver_cuda = nvidia_smi_driver_cuda()
        if driver_cuda:
            compatible_index = pytorch_cu_index_for(driver_cuda)
            print_kv("driver-supported CUDA max", driver_cuda)
            if compatible_index:
                print(
                    "Observation: a CUDA-enabled PyTorch build may be possible "
                    f"for this driver; highest known family here is {cuda_label(compatible_index)}."
                )
            else:
                print("Observation: no known PyTorch CUDA wheel index fits this driver version.")

        print("Result: CPU installation may proceed, but GPU acceleration will not be used.")
        print(
            "Expected: use a CUDA-enabled PyTorch build only if GPU support is required; "
            "then install a CUDA Toolkit / nvcc with the same torch.version.cuda X.Y."
        )
        return 0

    print_status("PASS", "CPU-only environment detected.")
    print("Result: no NVIDIA GPU runtime was detected; CPU installation path is valid.")
    print(
        "Required for this path: CPU-only PyTorch "
        f">= {version_text(MIN_TORCH_VERSION)}; CUDA Toolkit / nvcc is not required."
    )
    return 0


def main() -> int:
    root = repo_root()
    torch_info = detect_torch()
    nvccs = detect_nvccs()
    has_nvidia_smi, smi_output = detect_nvidia_smi()

    print("NssMPClib environment check")
    print(f"Repository: {root}")

    print_section("Environment")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME') or '<unset>'}")
    print(f"TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST') or '<unset>'}")

    print_section("PyTorch")
    if torch_info.installed:
        print(f"torch: {torch_info.version}")
        print(f"torch.version.cuda: {torch_info.cuda_version or '<cpu-only>'}")
        print(f"torch.cuda.is_available(): {torch_info.cuda_available}")
        if torch_info.devices:
            for idx, name, cap in torch_info.devices:
                print(f"GPU {idx}: {name} (sm_{cap.replace('.', '')})")
        if torch_info.error:
            print(f"warning: {torch_info.error}")
    else:
        print("torch: not installed")
        if torch_info.error:
            print(f"import error: {torch_info.error}")

    print_section("CUDA Toolchain")
    if nvccs:
        for info in nvccs:
            print(f"nvcc: {info.path} (release {info.release or 'unknown'})")
    else:
        print("nvcc: not found")

    if has_nvidia_smi:
        print("nvidia-smi: found")
        if smi_output:
            for line in smi_output.splitlines():
                print(f"  {line}")

        driver_cuda = nvidia_smi_driver_cuda()
        if driver_cuda:
            print(f"driver-supported CUDA (max): {driver_cuda}")

        caps = nvidia_smi_compute_caps()
        if caps:
            print("compute capability: " + ", ".join(caps))
    else:
        print("nvidia-smi: not found")

    print_section("Submodules")
    missing = submodule_status(root)
    if missing:
        print("missing: " + ", ".join(missing))
    else:
        print("cutlass: ok")
        print("csprng: ok")

    print_section("Build dependencies")
    for name in ("setuptools", "wheel"):
        version = package_version(name)
        if version is None:
            print(f"{name}: missing")
        elif name == "setuptools":
            print(f"{name}: {version} (required >= {version_text(MIN_SETUPTOOLS_VERSION)})")
        else:
            print(f"{name}: {version} (required: installed)")

    has_compiler, _ = detect_cpp_compiler()
    print(f"C/C++ compiler: {'found' if has_compiler else 'not found'}")

    print_required_versions(torch_info, nvccs)

    status = diagnose(root, torch_info, nvccs)

    print("\nNote: this script only diagnoses the environment; it does not install or modify anything.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
