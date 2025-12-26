import os

from setuptools import setup, find_packages

path = os.path.dirname(os.path.abspath(__file__))

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
    install_requires=[
        f'torchcsprng @ file://localhost/{path}/csprng',
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
