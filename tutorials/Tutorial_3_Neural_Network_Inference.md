# NssMPC Library Tutorial - Neural Network Inference with Secret Sharing

## Overview
This tutorial demonstrates how to perform privacy-preserving neural network inference using NssMPC. We support both 2-party and 3-party scenarios. The model owner holds a pre-trained neural network model, and data owners hold private input data. Through secret sharing techniques, data owners can obtain inference results without revealing their input data, and the model owner's weights remain confidential.

## Prerequisites

```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import NssMPC.application.neural_network as nn
from NssMPC import PartyRuntime, SEMI_HONEST, HONEST_MAJORITY
from NssMPC.config import DEVICE, NN_path
from NssMPC.infra.utils.debug_utils import get_time
import time
```

## Important: Separate Execution
**All parties must run in completely separate Python processes/scripts.** This ensures data privacy and proper network communication.

---

## 2-Party Neural Network Inference

### Server Setup (Model Owner - Party 0)
```python
# File: server_2pc.py - MUST be run separately
import os
import torch
import NssMPC.application.neural_network as nn
from NssMPC.config import NN_path
from NssMPC import PartyRuntime, SEMI_HONEST

# Define your model architecture
class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # Define your model layers here
        pass
    
    def forward(self, x):
        # Define forward pass
        pass

if __name__ == '__main__':
    # Initialize server (Party 0)
    server = nn.party.PartyNeuralNetwork2PC(0, SEMI_HONEST)
    
    with PartyRuntime(server):
        server.online()  # Establish connection with client
        
        # Load pre-trained model
        net = AlexNet()
        net.load_state_dict(torch.load(NN_path / 'AlexNet_CIFAR10.pkl'))
        
        # Share model parameters with client
        shared_param, shared_param_for_other = nn.utils.share_model(net)
        server.send(shared_param_for_other)
        
        # Load shared parameters into model
        net = nn.utils.load_model(net, shared_param)
        
        # Get number of inference requests
        num = server.dummy_model(net)
        
        # Inference loop
        while num:
            # Receive shared data from client
            shared_data = server.recv()
            
            # Perform privacy-preserving inference
            server.inference(net, shared_data)
            
            num -= 1
    
    server.close()
```

### Client Setup (Data Owner - Party 1)
```python
# File: client_2pc.py - MUST be run separately
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import NssMPC.application.neural_network as nn
from NssMPC import PartyRuntime, SEMI_HONEST
from NssMPC.config import DEVICE, NN_path
from NssMPC.infra.utils.debug_utils import get_time

# Define the same model architecture as server
class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # Define your model layers here
        pass
    
    def forward(self, x):
        # Define forward pass
        pass

if __name__ == '__main__':
    # Initialize client (Party 1)
    client = nn.party.PartyNeuralNetwork2PC(1, SEMI_HONEST)
    
    with PartyRuntime(client):
        client.online()  # Establish connection with server
        
        # Prepare test dataset
        transform1 = transforms.Compose([transforms.ToTensor()])
        test_set = torchvision.datasets.CIFAR10(root=NN_path, train=False, download=True, transform=transform1)
        
        # Use subset of data for testing
        indices = list(range(1024))  # First 1024 samples
        subset_data = Subset(test_set, indices)
        test_loader = torch.utils.data.DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=0)
        
        # Initialize model structure
        net = AlexNet()
        
        # Receive shared model parameters from server
        shared_param = client.recv()
        
        # Get number of inference requests (sync with server)
        num = client.dummy_model(test_loader)
        
        # Load shared parameters into model
        net = nn.utils.load_model(net, shared_param)
        
        correct_total = 0
        total_total = 0
        
        # Process each data sample
        for data in test_loader:
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Share input data with server
            shared_data, shared_data_for_other = nn.utils.share_data(images)
            client.send(shared_data_for_other)
            
            # Perform privacy-preserving inference
            res = get_time(client.inference, net, shared_data)
            
            # Get predictions
            _, predicted = torch.max(res, 1)
            
            # Calculate accuracy
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            total_total += total
            correct_total += correct
            
            print('Accuracy on test images: {}%'.format(100 * correct / total))
        
        print('Overall accuracy on test images: {}%'.format(100 * correct_total / total_total))
    
    client.close()
```

---

## 3-Party Neural Network Inference

### Party 0 Setup (Model Owner)
```python
# File: party0_3pc.py - MUST be run separately
import time
import torch
import NssMPC.application.neural_network as nn
from NssMPC.infra.tensor import RingTensor
from NssMPC.application.neural_network.party import PartyNeuralNetWork3PC
from NssMPC.protocols.honest_majority_3pc.base import share
from NssMPC.runtime import SEMI_HONEST, PartyRuntime
from data.AlexNet.Alexnet import AlexNet

if __name__ == '__main__':
    # Initialize Party 0 (Model Owner)
    P0 = PartyNeuralNetWork3PC(0, SEMI_HONEST)  # Can also use HONEST_MAJORITY
    P0.online()
    
    with PartyRuntime(P0):
        # Prepare test input
        test_input = torch.randint(-10, 10, [1, 3, 32, 32]) * 1.0
        print("Test input:", test_input)
        
        # Load and test model locally
        net = AlexNet()
        test_output = net(test_input)
        print("Test output", test_output)
        
        print("Sharing model weights...")
        # Share model weights with other parties
        shared_param = nn.utils.share_model(net, share_type=32)
        local_param = shared_param[0]    # P0's share
        P1_param = shared_param[1]       # P1's share
        P2_param = shared_param[2]       # P2's share
        
        # Send shares to other parties
        P0.send(1, P1_param)
        P0.send(2, P2_param)
        
        print("Preprocessing...")
        # Initialize model for secure inference
        num = P0.dummy_model(net, test_input)
        net = nn.utils.load_model(net, local_param)
        
        print("Sharing input data...")
        # Share input data with other parties
        share_input = share(RingTensor.convert_to_ring(test_input), P0)
        print("Share input reconstructed:", share_input.restore().convert_to_real_field())
        
        # Perform multiple inference runs
        for i in range(10):
            st = time.time()
            output = net(share_input)
            et = time.time()
            print("Inference time cost:", et - st)
        
        print("Final output:", output.restore().convert_to_real_field())
```

### Party 1 Setup (Computation Party)
```python
# File: party1_3pc.py - MUST be run separately
import NssMPC.application.neural_network as nn
from NssMPC.application.neural_network.party import PartyNeuralNetWork3PC
from NssMPC.protocols.honest_majority_3pc.base import receive_share_from
from NssMPC.runtime import SEMI_HONEST, PartyRuntime
from data.AlexNet.Alexnet import AlexNet

if __name__ == '__main__':
    # Initialize Party 1 (must use same mode as Party 0)
    P1 = PartyNeuralNetWork3PC(1, SEMI_HONEST)  # Mode must match Party 0
    P1.online()
    
    with PartyRuntime(P1):
        # Initialize model structure
        net = AlexNet()
        
        print("Receiving model weights...")
        # Receive model weight share from Party 0
        local_param = P1.recv(0)
        
        print("Preprocessing...")
        # Initialize for secure inference
        num = P1.dummy_model()
        net = nn.utils.load_model(net, local_param)
        
        print("Receiving input data...")
        # Receive input data share from Party 0
        share_input = receive_share_from(0, P1)
        print("Share input reconstructed:", share_input.restore().convert_to_real_field())
        
        # Perform inference (same computation as other parties)
        for i in range(10):
            output = net(share_input)
        
        print("Final output:", output.restore().convert_to_real_field())
```

### Party 2 Setup (Computation Party)
```python
# File: party2_3pc.py - MUST be run separately
import NssMPC.application.neural_network as nn
from NssMPC.application.neural_network.party import PartyNeuralNetWork3PC
from NssMPC.protocols.honest_majority_3pc.base import receive_share_from
from NssMPC.runtime import PartyRuntime, SEMI_HONEST
from data.AlexNet.Alexnet import AlexNet

if __name__ == '__main__':
    # Initialize Party 2 (must use same mode as other parties)
    P2 = PartyNeuralNetWork3PC(2, SEMI_HONEST)  # Mode must match other parties
    P2.online()
    
    with PartyRuntime(P2):
        # Initialize model structure
        net = AlexNet()
        
        print("Receiving model weights...")
        # Receive model weight share from Party 0
        local_param = P2.recv(0)
        
        print("Preprocessing...")
        # Initialize for secure inference
        num = P2.dummy_model()
        net = nn.utils.load_model(net, local_param)
        
        print("Receiving input data...")
        # Receive input data share from Party 0
        share_input = receive_share_from(0, P2)
        print("Share input reconstructed:", share_input.restore().convert_to_real_field())
        
        # Perform inference (same computation as other parties)
        for i in range(10):
            output = net(share_input)
        
        print("Final output:", output.restore().convert_to_real_field())
```

---

## Key Components Explained

### 1. Model Sharing (`nn.utils.share_model`)

**2-Party:**
```python
shared_param, shared_param_for_other = nn.utils.share_model(net)
```

**3-Party:**
```python
shared_param = nn.utils.share_model(net, share_type=32)
# Returns: [P0_share, P1_share, P2_share]
```

### 2. Data Sharing

**2-Party:**
```python
shared_data, shared_data_for_other = nn.utils.share_data(images)
```

**3-Party:**
```python
import NssMPC
# Party 0 shares input
share_input = NssMPC.SecretTensor(tensor=test_input)

# Other parties receive shares
share_input = NssMPC.SecretTensor(src_id=0)  # Party 1 & 2
```

### 3. Privacy-Preserving Inference

All parties perform the same inference computation on their shares:
```python
# 2-Party
output = server.inference(net, shared_data)  # Server side
output = client.inference(net, shared_data)  # Client side
#or
output = net(share_input)

# 3-Party (all parties execute same code)
output = net(share_input)
```

---

## Key Points to Remember

### For 2-Party Setup:
1. **Separate Execution**: Server and client in separate processes
2. **Order Matters**: Start server before client
3. **Model Consistency**: Both parties use identical model architecture
4. **Data Privacy**: Server never sees client's raw input

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
1. **Terminal 1**: `python server_2pc.py`
2. **Terminal 2**: `python client_2pc.py`

### 3-Party Inference:
1. **Terminal 1**: `python party0_3pc.py` (Model Owner)
2. **Terminal 2**: `python party1_3pc.py` (Computation Party)
3. **Terminal 3**: `python party2_3pc.py` (Computation Party)

## Customization Guide

### Using Different Models
```python
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

# Load your custom model
net = CustomModel()
net.load_state_dict(torch.load('custom_model.pth'))
```

### Performance Optimization
```python
# Batch processing for efficiency
test_loader = torch.utils.data.DataLoader(
    subset_data, 
    batch_size=32,  # Increased batch size
    shuffle=False, 
    num_workers=4
)

# Timing measurements
import time
start_time = time.time()
output = net(share_input)
end_time = time.time()
print(f"Inference time: {end_time - start_time:.4f} seconds")
```

## Error Handling
1. **Connection Issues**: Ensure Party 0 starts first in 3-party setup
2. **Model Mismatch**: Verify identical model architecture on all parties
3. **Memory Issues**: Reduce batch size if out of memory
4. **Mode Consistency**: All parties must use same security mode

## Performance Considerations
1. **2-Party vs 3-Party**: 3-party provides stronger security but more overhead
2. **Batch Size**: Larger batches improve throughput
3. **Network Latency**: Consider network speed between parties
4. **Model Complexity**: Larger models require more computation

Choose 2-party for efficiency when only two entities are involved, or 3-party for stronger security guarantees with honest majority assumption.