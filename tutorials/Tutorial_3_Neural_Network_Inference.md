from torch.utils.data import DataLoader

# NssMPC Library Tutorial - Neural Network Inference with Secret Sharing

## Overview
This tutorial demonstrates how to perform privacy-preserving neural network inference using NssMPC. We support both 2-party and 3-party scenarios. The model owner holds a pre-trained neural network model, and data owners hold private input data. Through secret sharing techniques, data owners can obtain inference results without revealing their input data, and the model owner's weights remain confidential.

## Prerequisites

Before running the Neural Network Inference with NssMPClib, ensure you have completed the setup as described in
the [Getting Started Guide](Tutorial_1_Getting_Started.md) and have the NssMPC library installed.

You should also have access to the pre-trained model weights and the dataset for inference. In this tutorial, we will
use the AlexNet model trained on the CIFAR-10 dataset.

First of all, match the security model and neural network model you want to use by modifying the scripts below (e.g.,
changing from SEMI_HONEST to HONEST_MAJORITY, or replacing AlexNet with ResNet)
and run the script to generate Beaver triples for neural network inference:

```bash
python scripts/neural_network_beaver_generation.py
```

Then we can proceed with the tutorial.

## Important: Separate Execution
**All parties must run in completely separate Python processes/scripts.** This ensures data privacy and proper network communication.

---

## 2-Party Neural Network Inference

### Party 0 Setup (Model Owner)

```python
# File: party0_2pc.py - MUST be run separately
import torch

import nssmpc.application.neural_network as nn
from nssmpc import PartyRuntime, Party2PC, SEMI_HONEST
from nssmpc.config import NN_path
from data.AlexNet.Alexnet import AlexNet

if __name__ == '__main__':

    party = Party2PC(0, SEMI_HONEST)
    party.online()
    with PartyRuntime(party):

        plaintext_model = AlexNet()
        plaintext_model.load_state_dict(torch.load(NN_path + 'AlexNet_CIFAR10.pkl'))

        shared_param = nn.utils.share_model_param(model=plaintext_model)
        SecAlexNet = nn.utils.convert_model(AlexNet)
        ciphertext_model = SecAlexNet()
        ciphertext_model = nn.utils.load_shared_param(ciphertext_model, shared_param)
        shared_data_loader = nn.utils.SharedDataLoader(src_id=1)

        for data in shared_data_loader:
            secret_result = ciphertext_model(data)
            secret_result.recon(target_id=1)

    party.close()
```

### Party 1 Setup (Data Owner)

```python
# File: party1_2pc.py - MUST be run separately
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

import nssmpc.application.neural_network as nn
from nssmpc import PartyRuntime, SEMI_HONEST, Party2PC
from nssmpc.config import NN_path
from nssmpc.infra.utils.profiling import RuntimeTimer
from data.AlexNet.Alexnet import AlexNet

if __name__ == '__main__':
    party = Party2PC(2, SEMI_HONEST)
    party.online()

    with PartyRuntime(party):
        transform1 = transforms.Compose([transforms.ToTensor()])
        test_set = torchvision.datasets.CIFAR10(root=NN_path, train=False, download=True, transform=transform1)

        indices = list(range(1024))
        subset_data = Subset(test_set, indices)
        test_loader = torch.utils.data.DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=0)

        shared_param = nn.utils.share_model_param(src_id=0)
        SecAlexNet = nn.utils.convert_model(AlexNet)
        ciphertext_model = SecAlexNet()
        ciphertext_model = nn.utils.load_shared_param(ciphertext_model, shared_param)
        shared_data_loader = nn.utils.SharedDataLoader(data_loader=test_loader)

        correct_total = 0
        total_total = 0

        for data in shared_data_loader:
            correct = 0
            total = 0
            inputs, labels = data

            with RuntimeTimer(tag="Inference", enable_comm_stats=True):
                secret_result = ciphertext_model(inputs)

            plaintext_result = secret_result.recon(target_id=1).convert_to_real_field()

            _, predicted = torch.max(plaintext_result, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            total_total += total
            correct_total += correct

            print('Accuracy of the communication on test images:{}%'.format(100 * correct / total))

        print('Accuracy of the communication on test images:{}%'.format(100 * correct_total / total_total))

    party.close()
```

---

## 3-Party Neural Network Inference

### Party 0 Setup (Model Owner)

```python
# File: party0_3pc.py - MUST be run separately
import torch.utils.data

import nssmpc.application.neural_network as nn
from nssmpc import Party3PC, SEMI_HONEST, PartyRuntime, SecretTensor
from nssmpc.infra.utils.profiling import RuntimeTimer
from data.AlexNet.Alexnet import AlexNet

if __name__ == '__main__':
    P = Party3PC(0, SEMI_HONEST)
    P.online()
    with PartyRuntime(P):
        test_input = torch.randint(-10, 10, [1, 3, 32, 32]) * 1.0
        print("test_input:", test_input)
        plaintext_model = AlexNet()
        test_output = plaintext_model(test_input)
        print("test_output", test_output)
        # Share model parameters
        shared_param = nn.utils.share_model_param(model=plaintext_model)
        # Convert to secure model class
        SecAlexNet = nn.utils.convert_model(AlexNet)
        # Instantiate secure model
        ciphertext_model = SecAlexNet()
        # Load shared parameters
        net = nn.utils.load_shared_param(ciphertext_model, shared_param)
        # Share input data
        share_input = SecretTensor(tensor=test_input)
        # Inference and profiling
        with RuntimeTimer(enable_comm_stats=True):
            output = net(share_input)
        # Reconstruct output to Party 0 and print
        print("output", output.recon(target_id=0).convert_to_real_field())
    P.close()
```

### Party 1 Setup (Computation Party)

```python
# File: party1_3pc.py - MUST be run separately
import nssmpc.application.neural_network as nn
from nssmpc import Party3PC, SEMI_HONEST, PartyRuntime, SecretTensor
from data.AlexNet.Alexnet import AlexNet

if __name__ == '__main__':
    P = Party3PC(1, SEMI_HONEST)
    P.online()
    with PartyRuntime(P):
        # Receive weights
        local_param = nn.utils.share_model_param(src_id=0)
        # Convert to secure model class
        SecAlexNet = nn.utils.convert_model(AlexNet)
        # Instantiate secure model
        ciphertext_model = SecAlexNet()
        # Load weights
        ciphertext_model = nn.utils.load_shared_param(ciphertext_model, local_param)
        # Receive input
        share_input = SecretTensor(src_id=0)
        # Inference
        output = ciphertext_model(share_input)
        # Reconstruct output to Party 0
        output.recon(target_id=0)
    P.close()
```

### Party 2 Setup (Computation Party)

```python
# File: party2_3pc.py - MUST be run separately
import nssmpc.application.neural_network as nn
from nssmpc import Party3PC, SEMI_HONEST, PartyRuntime, SecretTensor
from data.AlexNet.Alexnet import AlexNet

if __name__ == '__main__':
    P = Party3PC(2, SEMI_HONEST)
    P.online()
    with PartyRuntime(P):
        # Receive weights
        local_param = nn.utils.share_model_param(src_id=0)
        # Convert to secure model class
        SecAlexNet = nn.utils.convert_model(AlexNet)
        # Instantiate secure model
        ciphertext_model = SecAlexNet()
        # Load weights
        ciphertext_model = nn.utils.load_shared_param(ciphertext_model, local_param)
        # Receive input
        share_input = SecretTensor(src_id=0)
        # Inference
        output = ciphertext_model(share_input)
        # Reconstruct output to Party 0
        output.recon(target_id=0)
    P.close()
```

---

## Key Components Explained

### 1. Model Conversion (`nn.utils.convert_model`)

To convert a standard PyTorch model to a secure model compatible with NssMPC:

```python
import nssmpc.application.neural_network as nn

SecModelClass = nn.utils.convert_model(PlaintextModelClass)  # e.g., AlexNet
ciphertext_model = SecModelClass()
```

Or build the model manually using NssMPC layers:

```python
from torch.nn import Module
import nssmpc.application.neural_network as nn
from nssmpc.application.neural_network.layers import SecConv2d, SecReLU, SecLinear


class CustomModel(Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = SecConv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = SecReLU()
        self.fc = SecLinear(16 * 32 * 32, 10)
        # Add more layers as needed

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

### 2. Model Sharing (`nn.utils.share_model_param`)

For the model owner to share model parameters with the data owner:

```python
import nssmpc.application.neural_network as nn

shared_param = nn.utils.share_model_param(model=plaintext_model)
```

For the data owner to receive shared parameters:

```python
import nssmpc.application.neural_network as nn

shared_param = nn.utils.share_model_param(src_id=MODEL_OWNER_ID)
```

### 3. Data Sharing

#### Single Input Sharing

For the data owner to share input data:

```python
from nssmpc import SecretTensor

share_input = SecretTensor(tensor=test_input)
```

For the model owner or other parties to receive shared input data:

```python
from nssmpc import SecretTensor

share_input = SecretTensor(src_id=DATA_OWNER_ID)  # Party 1 & 2
```

#### DataLoader Sharing

For sharing a DataLoader:

```python
import nssmpc.application.neural_network as nn

shared_data_loader = nn.utils.SharedDataLoader(data_loader=dataloader)  # dataloader from PyTorch
```

For receiving shared DataLoader:

```python
import nssmpc.application.neural_network as nn

shared_data_loader = nn.utils.SharedDataLoader(src_id=DATA_OWNER_ID)
```

Then iterate over `shared_data_loader` to get secret-shared batches.

### 4. Privacy-Preserving Inference

To perform inference on secret-shared data, just call the model as usual:

```python
secret_result = ciphertext_model(share_input)
```

---

## Key Points to Remember

### For 2-Party Setup:

1. **Separate Execution**: Parties in separate processes
2. **Model Consistency**: Both parties use identical model architecture
3. **Data Privacy**: Other participants besides the data owner NEVER see plaintext data

### For 3-Party Setup:
1. **Three Separate Processes**: Each party in its own script
2. **Consistent Mode**: All parties use same security mode (SEMI_HONEST or HONEST_MAJORITY)
3. **Model Structure**: All parties initialize same model architecture

### Security Guarantees:
- **Input Privacy**: Data owner's input remains private
- **Model Privacy**: Model owner's weights remain private
- **Output Privacy**: Only authorized parties learn inference results

## Execution Instructions

### 2-Party Inference:

1. **Terminal 1**: `python party0_2pc.py`
2. **Terminal 2**: `python party1_2pc.py`

### 3-Party Inference:
1. **Terminal 1**: `python party0_3pc.py` (Model Owner)
2. **Terminal 2**: `python party1_3pc.py` (Computation Party)
3. **Terminal 3**: `python party2_3pc.py` (Computation Party)

## Customization Guide

### Using Different Models

```python
import torch
import nssmpc.application.neural_network as nn
from nssmpc import PartyRuntime, SecretTensor


class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer1 = torch.nn.Linear(784, 256)
        self.layer2 = torch.nn.Linear(256, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


with PartyRuntime(party):
    # Receive shared parameters and load into custom model
    shared_param = nn.utils.share_model_param(src_id=0)

    net = CustomModel()
    ciphertext_model = nn.utils.convert_model(CustomModel)
    # or use Modules in nssmpc.application.neural_network.layers to build model manually

    # Load shared parameters
    nn.utils.load_shared_param(ciphertext_model, shared_param)

    # Share input data and perform inference as shown earlier
    ...
```

## Error Handling

1. **Connection Issues**: Ensure all parties are online and reachable
2. **Model Mismatch**: Verify the same architecture is used
3. **Memory Issues**: Reduce batch size if out of memory
4. **Mode Consistency**: All parties must use same security mode

## Performance Considerations
1. **2-Party vs 3-Party**: 3-party provides stronger security but more overhead
2. **Batch Size**: Larger batches improve throughput
3. **Network Latency**: Consider network speed between parties
4. **Model Complexity**: Larger models require more computation

Choose 2-party for efficiency when only two entities are involved, or 3-party for stronger security guarantees with honest majority assumption.