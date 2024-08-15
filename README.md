# NssMPClib

## Introduction

This project is a secure multi-party computation library that designs and implements privacy-preserving computation
protocols based on arithmetic secret sharing and function secret sharing. It also utilizes these protocols to implement
the application of privacy-preserving machine learning, specifically privacy-preserving neural network inference.

## Installation tutorial

This project requires PyTorch>=1.8.0, and it is recommended to use PyTorch==1.8.0. Other dependencies are listed in the
./requirements.txt file. You can install the project dependencies by executing the following command:

```bash
pip install -r requirements.txt
```

## Getting start

All test codes needs to be run in the project root directory, as shown in the following example:

```
# Open two terminals and input the following code in each terminal:
python debug/application/neural_network/2pc/neural_network_server.py
python debug/application/neural_network/2pc/neural_network_client.py
```

If you cannot start using the above command, try adding the following code to the beginning of the corresponding test
file:

```python
import sys

sys.path.append('/path/to/the/directory/nssmpclib')
print(sys.path)
```

For instructions on how to use the library for a privacy application, please refer to the tutorials in the pack '
tutorials', which are presented as a Jupyter notebook, so please install the following in your conda environment:

```bash
conda install ipython jupyter
```

1.Tutorial_0_Before_Starting.ipynb - Before starting the tutorial, this notebook provides an introduction to the
configuration information and auxiliary parameters required for computations in the library.  
2.Tutorial_1_Ring_Tensor.ipynb - This tutorial introduces the basic data type `RingTensor` in the library. It
demonstrates how to perform basic operations using RingTensor.  
3.Tutorial_2_Arithmetic_Secret_Sharing.ipynb - This tutorial explains the basic data type, `ArithmeticSharedRingTensor`,
used for secure multi-party computation in the library. It shows how to perform basic operations using
ArithmeticSharedRingTensor through arithmetic secret sharing techniques that distribute data into two shares for two
participating parties.  
4.Tutorial_3_Generate_Beaver_Triples_by_HE.ipynb - This tutorial explains how to generate Beaver triples using
homomorphic encryption.  
5.Tutorial_4_Function_Secret_Sharing.ipynb - This tutorial covers function secret sharing in the library. It introduces
distributed point functions, distributed comparison functions, and the process of generating and evaluating distributed
interval containment functions.  
6.Tutorial_5_Neural_Network_Inference.ipynb - This tutorial explains how to implement privacy-preserving neural network
inference in this library.

## Architecture

- [application](https://github.com/XidianNSS/NssMPClib/tree/main/application)  
  The application package contains applications implemented using the functionalities of NssMPClib. Currently, it
  supports automatic conversion of plaintext cipher models and privacy-preserving neural network inference.
- [common](https://github.com/XidianNSS/NssMPClib/tree/main/common)   
  The common package includes general utilities and the basic data structures used by this lib, such as network
  communication, random number generators, and other tools.
- [config](https://github.com/XidianNSS/NssMPClib/tree/main/config)   
  The config package includes the basic configuration and network configuration of NssMPClib.
- [crypto](https://github.com/XidianNSS/NssMPClib/tree/main/crypto)   
  The crypto package is the core of the lib and includes the privacy computation primitives and protocols.
- [data](https://github.com/XidianNSS/NssMPClib/tree/main/data)  
  The data package includes the auxiliary parameters used by the library for computation and the models and datasets
  used for privacy-preserving neural network inference.
- [debug](https://github.com/XidianNSS/NssMPClib/tree/main/debug)    
  The debug package includes the test code for the lib.
- [model](https://github.com/XidianNSS/NssMPClib/tree/main/model)  
  The model package includes system models and threat models used by the lib, such as the client-server model under
  semi-honest assumptions.
- [tutorials](https://github.com/XidianNSS/NssMPClib/tree/main/tutorials)  
  The tutorials package contains the usage tutorials of NssMPClib.

## Notice
This branch only supports secure two-party computation and privacy-preserving convolutional neural network inference. Secure three-party computation, privacy-preserving large language model inference, and other features are available in the **[beta](https://github.com/XidianNSS/NssMPClib/tree/beta)** branch.

## Maintainers

This project is maintained by the NSS Team at Xidian University.

## License

NssMPClib is based on the MIT license, as described in LICENSE.

## Contact us
xidiannss@gmail.com
