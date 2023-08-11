import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 3
batch_size = 100

learning_rate = 0.001

# MNIST数据集
train_dataset = torchvision.datasets.CIFAR100(root='./data',
                                              train=True,
                                              transform=transforms.ToTensor(),
                                              download=True)

test_dataset = torchvision.datasets.CIFAR100(root='./data',
                                             train=False,
                                             transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

# 初始化模型
net = Net(100).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# 训练网络
total_step = len(train_loader)
for epoch in range(num_epochs):
    net.train()
    acc = 0.0
    sum = 0.0
    loss_sum = 0

    for batch, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = net(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc += torch.sum(torch.argmax(output, dim=1) == target).item()
        sum += len(target)
        loss_sum += loss.item()

        if (batch + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, batch+1, total_step, loss.item()))

    # 计算准确率和损失函数的均值
    train_accuracy = 100 * acc / sum
    train_loss = loss_sum / (batch + 1)

    # 创建存储训练记录的列表
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 在训练过程中，还需要计算测试集上的准确率和损失函数的均值
    net.eval()
    test_acc = 0.0
    test_sum = 0.0
    test_loss_sum = 0

    for batch, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = net(data)
        test_loss = criterion(output, target)
        test_acc += torch.sum(torch.argmax(output, dim=1) == target).item()
        test_sum += len(target)
        test_loss_sum += test_loss.item()

    test_accuracy = 100 * test_acc / test_sum
    test_loss = test_loss_sum / (batch + 1)

    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print('Epoch %d/%d: train acc: %.2f%%, train loss: %.4f, test acc: %.2f%%, test loss: %.4f' %
        (epoch+1, num_epochs, train_accuracy, train_loss, test_accuracy, test_loss))


plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Train')
plt.plot(range(1, num_epochs+1), test_losses, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train')
plt.plot(range(1, num_epochs+1), test_accuracies, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')
plt.legend()
plt.show()

