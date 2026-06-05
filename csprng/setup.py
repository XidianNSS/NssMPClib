import glob
import os
import re
import shlex
import shutil
import site
import subprocess
import sys

import torch
from setuptools import Command, find_packages, setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


YES_VALUES = ("1", "true", "yes", "on")
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


def _ensure_torch_version():
    version = getattr(torch, "__version__", None)
    if _version_at_least(version, MIN_TORCH_VERSION):
        return

    raise RuntimeError(
        "PyTorch is too old for bundled torchcsprng.\n"
        f"Required: torch >= {_version_text(MIN_TORCH_VERSION)}.\n"
        f"Current torch: {version or '<unknown>'}."
    )


def _nvcc_release(cuda_home):
    if not cuda_home:
        return None

    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.isfile(nvcc):
        return None

    try:
        out = subprocess.check_output(
            [nvcc, "--version"],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", "ignore")
    except (OSError, subprocess.CalledProcessError):
        return None

    match = re.search(r"release\s+(\d+\.\d+)", out)
    return match.group(1) if match else None


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
    """Return possible CUDA roots.

    Supports:
      - system CUDA: /usr/local/cuda, /usr/local/cuda-*
      - conda CUDA: $CONDA_PREFIX
      - nvcc discovered on PATH
      - manually configured CUDA_HOME / CUDA_PATH
    """
    candidates = []

    for env_name in ("CUDA_HOME", "CUDA_PATH", "CONDA_PREFIX"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(value)

    if cpp_extension.CUDA_HOME:
        candidates.append(cpp_extension.CUDA_HOME)

    path_nvcc = shutil.which("nvcc")
    if path_nvcc:
        candidates.append(os.path.dirname(os.path.dirname(path_nvcc)))

    candidates.append("/usr/local/cuda")
    candidates.extend(sorted(glob.glob("/usr/local/cuda-*"), reverse=True))

    result = []
    seen = set()

    for candidate in candidates:
        if not candidate:
            continue

        candidate = os.path.abspath(candidate)

        if candidate in seen:
            continue

        seen.add(candidate)
        result.append(candidate)

    return result


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
        pattern = os.path.join(conda_prefix, "lib", "python*", "site-packages")
        dirs.extend(glob.glob(pattern))

    return _dedupe_existing_paths(dirs)


def _cuda_include_dirs():
    """Return CUDA include directories.

    Handles:
      - /usr/local/cuda*/include
      - $CONDA_PREFIX/include
      - $CONDA_PREFIX/targets/x86_64-linux/include
      - pip NVIDIA wheels: site-packages/nvidia/*/include
    """
    candidates = []

    for root in _cuda_home_candidates():
        candidates.extend(
            [
                os.path.join(root, "include"),
                os.path.join(root, "targets", "x86_64-linux", "include"),
            ]
        )

    for site_dir in _python_site_dirs():
        candidates.extend(glob.glob(os.path.join(site_dir, "nvidia", "*", "include")))

    return _dedupe_existing_paths(candidates)


def _cuda_library_dirs():
    """Return CUDA library directories.

    Handles:
      - /usr/local/cuda*/lib64
      - $CONDA_PREFIX/lib
      - $CONDA_PREFIX/targets/x86_64-linux/lib
      - pip NVIDIA wheels: site-packages/nvidia/*/lib
    """
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


def _find_header(header_name, include_dirs):
    for include_dir in include_dirs:
        candidate = os.path.join(include_dir, header_name)
        if os.path.exists(candidate):
            return candidate
    return None


def _ensure_cuda_headers(include_dirs):
    required_headers = [
        "cuda_runtime.h",
        "cublas_v2.h",
    ]

    missing = []
    found = {}

    for header in required_headers:
        path = _find_header(header, include_dirs)
        if path:
            found[header] = path
        else:
            missing.append(header)

    if missing:
        lines = [
            "Error: CUDA development headers are missing.",
            "",
            "Missing headers:",
        ]

        for header in missing:
            lines.append(f"  - {header}")

        lines.extend(
            [
                "",
                "Checked include directories:",
            ]
        )

        if include_dirs:
            for include_dir in include_dirs:
                lines.append(f"  - {include_dir}")
        else:
            lines.append("  <none>")

        lines.extend(
            [
                "",
                "This usually means nvcc is available, but CUDA development headers",
                "are not installed or are not visible to the build system.",
                "",
                "For pip/PyTorch CUDA wheels, headers may live under:",
                "  site-packages/nvidia/*/include",
                "",
                "For conda/miniforge CUDA environments, headers may live under:",
                "  $CONDA_PREFIX/targets/x86_64-linux/include",
                "",
                "To skip CUDA support for torchcsprng, set:",
                "  NSSMPC_SKIP_CSPRNG_CUDA=1",
            ]
        )

        raise RuntimeError("\n".join(lines))

    print("Detected CUDA headers:")
    for header, path in found.items():
        print(f"  {header}: {path}")


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


def _cuda_compiler_issue(torch_cuda):
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
        getattr(cpp_extension, "CUDA_CLANG_VERSIONS", {})
        if kind == "clang"
        else getattr(cpp_extension, "CUDA_GCC_VERSIONS", {})
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


def _ensure_cuda_compiler_compatible(torch_cuda):
    issue = _cuda_compiler_issue(torch_cuda)
    if issue:
        raise RuntimeError(
            "C++ compiler version is incompatible with this CUDA/PyTorch build.\n"
            f"Reason: {issue}\n"
            "Required: use a host C++ compiler version within PyTorch's CUDA compiler bounds, "
            "or set NSSMPC_SKIP_CSPRNG_CUDA=1 for an intentional CPU-only torchcsprng build."
        )


def _auto_set_cuda_home(torch_cuda):
    """Align CUDA_HOME to torch.version.cuda when possible.

    This supports both system CUDA and conda/miniforge CUDA layouts.
    """
    if not torch_cuda:
        return True

    current = os.environ.get("CUDA_HOME")
    if current and _nvcc_release(current) == torch_cuda:
        os.environ.setdefault("CUDA_PATH", current)
        return True

    candidates = _cuda_home_candidates()

    for candidate in candidates:
        if _nvcc_release(candidate) == torch_cuda:
            os.environ["CUDA_HOME"] = candidate
            os.environ.setdefault("CUDA_PATH", candidate)
            print(
                f"Notice: auto-set CUDA_HOME={candidate} "
                f"(matches torch.version.cuda={torch_cuda})"
            )
            return True

    return False


_ensure_torch_version()


version = open("version.txt", "r").read().strip()
sha = "Unknown"
package_name = "torchcsprng"
cwd = os.path.dirname(os.path.abspath(__file__))

try:
    sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .decode("ascii")
        .strip()
    )
except Exception:
    pass

if os.getenv("BUILD_VERSION"):
    version = os.getenv("BUILD_VERSION")
elif sha != "Unknown":
    version += "+" + sha[:7]

print(f"Building wheel {package_name}-{version}")


def write_version_file():
    version_path = os.path.join(cwd, "torchcsprng", "version.py")
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))


write_version_file()

with open("README.md", "r") as fh:
    long_description = fh.read()


def append_flags(flags, flags_to_append):
    for flag in flags_to_append:
        if flag not in flags:
            flags.append(flag)
    return flags


def get_extensions():
    skip_cuda = _env_enabled("NSSMPC_SKIP_CSPRNG_CUDA")

    build_cuda = not skip_cuda and (
        torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1"
    )

    if skip_cuda:
        print(
            "Notice: NSSMPC_SKIP_CSPRNG_CUDA is set; "
            "building torchcsprng without CUDA support."
        )

    module_name = "torchcsprng"
    extensions_dir = os.path.join(cwd, module_name, "csrc")

    openmp = "ATen parallel backend: OpenMP" in torch.__config__.parallel_info()

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))

    sources = main_file + source_cpu
    extension = CppExtension
    define_macros = []

    cxx_flags = os.getenv("CXX_FLAGS", "")
    if cxx_flags == "":
        cxx_flags = []
    else:
        cxx_flags = cxx_flags.split(" ")

    if openmp:
        if sys.platform == "linux":
            cxx_flags = append_flags(cxx_flags, ["-fopenmp"])
        elif sys.platform == "win32":
            cxx_flags = append_flags(cxx_flags, ["/openmp"])

    include_dirs = []
    library_dirs = []

    if build_cuda:
        if not _auto_set_cuda_home(torch.version.cuda):
            raise RuntimeError(
                "CUDA PyTorch is installed, but no matching CUDA Toolkit / nvcc "
                "was found for bundled torchcsprng.\n"
                f"Required: CUDA Toolkit / nvcc {torch.version.cuda}, "
                "matching torch.version.cuda."
            )

        if os.environ.get("CUDA_HOME"):
            cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]

        extension = CUDAExtension

        source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
        sources += source_cuda

        define_macros += [("WITH_CUDA", None)]

        include_dirs = _cuda_include_dirs()
        library_dirs = _cuda_library_dirs()

        _ensure_cuda_headers(include_dirs)
        _ensure_cuda_compiler_compatible(torch.version.cuda)

        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags == "":
            nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(" ")

        for include_dir in include_dirs:
            nvcc_flags = append_flags(nvcc_flags, [f"-I{include_dir}"])

        nvcc_flags = append_flags(
            nvcc_flags,
            [
                "--expt-extended-lambda",
                "-Xcompiler",
            ],
        )

        extra_compile_args = {
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        }

        print("Building torchcsprng with CUDA support.")
        print("CUDA_HOME:", os.environ.get("CUDA_HOME") or "<unset>")

        print("CUDA include dirs:")
        for include_dir in include_dirs:
            print(f"  - {include_dir}")

        print("CUDA library dirs:")
        for library_dir in library_dirs:
            print(f"  - {library_dir}")

    else:
        extra_compile_args = {
            "cxx": cxx_flags,
        }

    ext_modules = [
        extension(
            module_name + "._C",
            sources,
            define_macros=define_macros,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


class fast_install(Command):
    description = "Custom install command that cleans project and installs wheel"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system("python setup.py clean")
        os.system("python setup.py bdist_wheel")
        os.system(f"pip install {glob.glob('./dist/*.whl')[0]} --force-reinstall --no-deps")


class clean(Command):
    description = "Custom clean command that cleans project based on .gitignore rules"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        with open(".gitignore", "r") as f:
            ignores = f.read()

        start_deleting = False

        for wildcard in filter(None, ignores.split("\n")):
            if (
                wildcard
                == "# do not change or delete this comment - `python setup.py clean` deletes everything after this line"
            ):
                start_deleting = True

            if not start_deleting:
                continue

            for filename in glob.glob(wildcard, recursive=True):
                try:
                    os.remove(filename)
                    print(f"Removed file: {filename}")
                except OSError:
                    shutil.rmtree(filename, ignore_errors=True)
                    print(f"Removed directory: {filename}")


setup(
    name=package_name,
    version=version,
    author="Pavel Belevich",
    author_email="pbelevich@fb.com",
    url="https://github.com/pytorch/csprng",
    description="Cryptographically secure pseudorandom number generators for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD-3",
    packages=find_packages(exclude=("test",)),
    package_data={"": ["*.pyi"]},
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires="torch>=2.5.0",
    ext_modules=get_extensions(),
    test_suite="test",
    cmdclass={
        "fast_install": fast_install,
        "build_ext": BuildExtension,
        "clean": clean,
    },
)
