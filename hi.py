import torch
import torchvision
from matplotlib import pyplot as plt
import torch.optim as optim
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 设定超参数
batch_size = 64
num_epochs = 5


learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []



class ResNet_GRU(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5, freeze_resnet=True):
        super(ResNet_GRU, self).__init__()
        self.resnet34 = models.resnet34(pretrained=True)
        if freeze_resnet:
            for param in self.resnet34.fc:
                param.requires_grad = False

        self.gru = nn.GRU(1000, 256, 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet34(x)
        x, _ = self.gru(x.view(x.size(0), 1, -1))
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

def main():
    '''transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(), # 数据增强：水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])'''

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    model = ResNet_GRU(freeze_resnet=True)
    #
    #model.load_state_dict(torch.load('method1.pth'))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # 训练模型
    model.train()
    for epoch in range(num_epochs):
        acc = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)  # 获取最大值对应的索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},ass:{}'
                      .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(),correct / total))


            train_losses.append(loss.item())
            train_accuracies.append( correct / total)

    # 测试模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_losses.append(loss)
        test_accuracies.append(100 * correct / total)

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

        torch.save(model.state_dict(),"way1.pth")


if __name__ == '__main__':
    main()


