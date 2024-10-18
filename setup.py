import os

from setuptools import setup, find_packages

path = os.path.dirname(os.path.abspath(__file__))

setup(
    name="NssMPClib",
    version="1.0",
    author="XDU_NSS",
    author_email="",
    description="NssMPClib项目是一个通用的安全多方计算库，设计并实现了一系列基于算术秘密共享（Arithmetic Secret Sharing，ASS）和函数秘密共享（Function Secret Sharing，FSS）的隐私保护计算协议，并实现了神经网络密态推理等隐私保护机器学习应用。",
    url="https://gitcode.com/openHiTLS/NssMPClib",
    license="MIT",
    packages=find_packages(where='.NssMPC'),
    include_package_data=True,
    install_requires=[f'torchcsprng @ file://localhost/{path}/csprng'],
)
