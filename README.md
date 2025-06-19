# NssMPClib
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/XidianNSS/NssMPClib)

## Introduction

This project is a secure multi-party computation library that designs and implements privacy-preserving computation
protocols based on arithmetic secret sharing and function secret sharing.
It also utilizes these protocols to implement the application of privacy-preserving machine learning, specifically
privacy-preserving neural network inference.

## Installation

This project requires PyTorch>=1.8.0, and it is recommended to use PyTorch==2.3.0.
You can install this project by executing the following command:

```bash
pip install -e .
```

If the external dependency csprng cannot be compiled and installed, it may be due to a lack of a c++ compiler or cuda
toolkit.

## Getting start

For instructions on how to use the library for a privacy application, please refer to the tutorials in the pack 
`tutorials`, which are presented as a Jupyter notebook, so please install the following in your conda environment:

```bash
conda install ipython jupyter
```

1. `Tutorial_0_Before_Starting.ipynb` - Before starting the tutorial, this notebook provides an introduction to the
   configuration information and auxiliary parameters required for computations in the library.
2. `Tutorial_1_Ring_Tensor.ipynb` - This tutorial introduces the basic data type `RingTensor` in the library. It
   demonstrates how to perform basic operations using RingTensor.
3. `Tutorial_2_Arithmetic_Secret_Sharing.ipynb` - This tutorial explains the basic data
   type, `ArithmeticSharedRingTensor`, used for secure multi-party computation in the library. It shows how to perform
   basic operations using ArithmeticSharedRingTensor through arithmetic secret sharing techniques that distribute data
   into two shares for two
   participating parties.
4. `Tutorial_3_Replicated_Secret_sharing.ipynb` - This tutorial introduces the basic data
   type, `ReplicatedSecretSharing`,
   used for secure 3-party computation in the library. It distributes data into multiple shares for multiple
   participating parties using replicated secret sharing techniques. The basic operations of `ReplicatedSecretSharing`
   are similar to those of `ArithmeticSecretSharing`, and tutorials on related basic operations are coming soon.
5. `Tutorial_4_Generate_Beaver_Triples_by_HE.ipynb` - This tutorial explains how to generate Beaver triples using
   homomorphic encryption.
6. `Tutorial_5_Parameter.ipynb` - This tutorial explains how to design, implement, generate, and use the auxiliary
   parameters required for secure multi-party computation in the library.
7. `Tutorial_6_Function_Secret_Sharing.ipynb` - This tutorial covers function secret sharing in the library. It
   introduces
   distributed point functions, distributed comparison functions, and the process of generating and evaluating
   distributed
   interval containment functions.
8. `Tutorial_7_Neural_Network_Inference.ipynb` - This tutorial explains how to implement privacy-preserving neural
   network inference in this library.

## Architecture

- **csprng**  
    Customized torchcsprng source code, as an external dependency of the library.
- **data**
    Used to store the plaintext model structure code for privacy-preserving neural network inference. Other related data
    required by other applications can also be placed in this folder.
- **debug**  
    Test code for this project.
- **NssMPC**  
  Library main source code.
    - application  
      The application package contains applications implemented using the functionalities of NssMPClib. Currently, it
      supports automatic conversion of plaintext cipher models and privacy-preserving neural network inference.
    - common  
      The common package includes general utilities and the basic data structures used by this lib, such as network
      communication, random number generators, and other tools.
    - config  
      The config package includes the basic configuration and network configuration of NssMPClib.
    - crypto  
      The crypto package is the core of the lib and includes the privacy computation primitives and protocols.
    - secure_model  
      The secure_model package includes system models and threat models used by the lib, such as the client-server model
      under semi-honest assumptions.
- **tutorials**  
  The tutorials package contains the usage tutorials of NssMPClib.

## API Documentation

See [NssMPClib Documentation](https://www.xidiannss.com/doc/NssMPClib/index.html).

## Maintainers

This project is maintained by the XDU NSS lab.

## License

NssMPClib is based on the MIT license, as described in LICENSE.

## Contact us

email: xidiannss@gmail.com
