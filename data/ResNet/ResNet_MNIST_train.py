# ResNet & MNIST

import time

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from data.ResNet.ResNet import resnet50

transform = transforms.Compose([
    transforms.ToTensor(),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_set = torchvision.datasets.MNIST(root='data', train=True, download=True,
                                       transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)

test_set = torchvision.datasets.MNIST(root='data', train=False, download=True,
                                      transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

net = resnet50()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=1e-3)

net.to(device)

print("Start Training!")

num_epochs = 1

for epoch in range(num_epochs):
    running_loss = 0
    batch_size = 10

    for i, data in enumerate(train_loader):
        print(i)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('[%d, %5d] loss:%.4f' % (epoch + 1, (i + 1) * 100, loss.item()))

print("Finished Training")

net.eval()
torch.save(net.state_dict(), 'data/ResNet/ResNet50_MNIST.pkl')

net.load_state_dict(torch.load('data/ResNet/ResNet50_MNIST.pkl'))
start_time = time.time()

with torch.no_grad():
    total_correct = 0
    total_total = 0

    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        out = net(images)
        _, predicted = torch.max(out.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        total_total += total
        total_correct += correct
        print('Accuracy of the network on the 100 test images:{}%'.format(100 * correct / total))
end_time = time.time()
print("time: ", end_time - start_time)
print('Accuracy of the network on the 10000 test images:{}%'.format(100 * total_correct / total_total))
