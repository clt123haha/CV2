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
num_epochs = 1

learning_rate = 0.001
train_loss = []
test_loss =  0.0
train_acc = []
test_acc = []
batch_x = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ResNet_GRU(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5, freeze_resnet=True):
        super(ResNet_GRU, self).__init__()
        self.resnet34 = models.resnet34(pretrained=True)
        if freeze_resnet:
            for param in self.resnet34.fc.parameters():
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
    batch = 1
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
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # 训练模型
    model.train()
    for epoch in range(num_epochs):
        acc = 0.0
        correct = 0
        total = 0
        epoch_loss = 0.0  # 用于记录整个epoch的损失值
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:
                train_acc.append(100 * correct / total)

                # 追加每个batch的epoch到batch_x列表
                batch_x.append(batch)
                batch += 1

                # 将每个epoch的损失追加到train_loss列表中
                train_loss.append(epoch_loss / len(train_loader))
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), 100 * correct / total))

            epoch_loss += loss.item()  # 累积每个batch的损失





    # 测试模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        test_loss = 0.0  # 移动到循环外部
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 累积测14试损失
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)  # 计算平均测试损失
        test_accuracy = 100 * correct / total
        test_acc.append(test_accuracy)

        print('Test Accuracy of the model on the 10000 test images: {:.2f}%'.format(test_accuracy))
    #torch.save(model.state_dict(), "way1.pth")

    print(test_accuracy)
    print(test_loss)
    print(train_acc)
    print(train_loss)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(batch_x, train_loss, color='tab:red', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # 创建一个共享 x 轴但独立 y 轴的副图
    ax2 = ax1.twinx()

    # 绘制训练准确率曲线（使用 ax2）
    ax2.set_ylabel('Accuracy (%)', color='tab:blue')
    ax2.plot(batch_x, train_acc, color='tab:blue', label='Train Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # 在ax1上绘制测试损失和测试准确率的目标值水平线
    target_loss_value = test_loss
    ax1.axhline(y=target_loss_value, color='r', linestyle='--', label='Test Loss')
    target_acc_value = test_acc[0]
    ax2.axhline(y=target_acc_value, color='tab:blue', linestyle='--', label='Test Accuracy')

    # 显示图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

    # 设置标题
    fig.suptitle('Training/Test Metrics')

    # 显示图形
    plt.show()



if __name__ == '__main__':
    main()


