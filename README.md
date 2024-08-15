# NssMPClib

## Introduction

This project is a secure multi-party computation library that designs and implements privacy-preserving computation
protocols based on arithmetic secret sharing and function secret sharing. It also utilizes these protocols to implement
the application of privacy-preserving machine learning, specifically privacy-preserving neural network inference.

This branch has more features and is more flexible than the master branch. However, it
may have some bugs. If you find any bugs, please report them to us.

## Installation

This library needs pytorch and prng support, pytorch installation refer to pytorch official website, prng needs C++
compilation environment support, the installation command is as follows:

```shell
# go to prng directory
cd ./PRNG  
# install prng
python setup.py fast_install
```

## Usage

Compared to the master branch, we have changed the structure of the project and the naming of some interfaces, but there
is not much difference in the use of it, you can still refer to the README.md and tutorials of the master branch for
more details on how to use it.

Before you can use the library for secure multi-party computations, you need to generate auxiliary parameters, the
relevant generation code is in `. /offline_parameter_generation`, run the following command to generate the auxiliary
parameters:

```shell
python offline_parameter_generation.py
```

Unlike the master branch, the configuration file and all auxiliary parameters will be generated under the system path (
Windows: `C:\Users\[UserName]\.NssMPClib`; Linux: `/home/.NssMPClib`), so there is no need to modify the file's running
path.

Tutorials and documentation are still being developed, so stay tuned.

## Maintainers

This project is maintained by the NSS Team at Xidian University.

## License

NssMPClib is based on the MIT license, as described in LICENSE.

## Connect us

xidiannss@gmail.com
