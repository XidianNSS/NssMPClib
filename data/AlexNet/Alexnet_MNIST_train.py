# AlexNet & MNIST
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from NssMPC.config import NN_path
from data.AlexNet.Alexnet import AlexNet

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),

])

transform1 = transforms.Compose([
    transforms.ToTensor()
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainset = torchvision.datasets.CIFAR10(root=NN_path, train=True, download=True,
                                      transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True,
                                          num_workers=0)

testset = torchvision.datasets.CIFAR10(root=NN_path,
                                     train=False, download=True,
                                     transform=transform1)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)

net = AlexNet()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.5)

net.to(device)

print("Start Training!")

num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0
    batch_size = 100

    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('[%d, %5d] loss:%.4f' % (epoch + 1, (i + 1) * 100, loss.item()))

print("Finished Training")

if not os.path.exists(NN_path):
    os.makedirs(NN_path)
torch.save(net.state_dict(), NN_path + '/AlexNet_MNIST.pkl')

net.load_state_dict(torch.load(NN_path + '/AlexNet_MNIST.pkl'))
start_time = time.time()

with torch.no_grad():
    total_correct = 0
    total_total = 0

    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        out = net(images)
        _, predicted = torch.max(out.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        total_total += total
        total_correct += correct

end_time = time.time()
print("time", end_time - start_time)
print('Accuracy of the network on the 100 test images:{}%'.format(100 * total_correct / total_total))
