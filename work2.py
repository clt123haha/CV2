import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet50, resnet34
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
#transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
transforms.Resize(224),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

# 超参数设置
num_epochs = 2

batch_size = 64
learning_rate = 0.001

train_loss = []
test_loss = []
train_acc = []
test_acc = []
batch_x = []
batch = 1

# 加载并预处理CIFAR-100数据集
train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 加载ResNet-50模型
model = resnet34(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 100)


for p in model.modules():
    if p._get_name()!='Linear':
        p.requires_grad_=False



model.load_state_dict(torch.load('work1.pth')) #装载上传训练的参数

# 将模型移动到设备上
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

model.train()
# 训练模型
for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
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

    train_accuracy = correct / total
    print('Epoch [{}/{}], Training Loss: {:.4f}, Training Accuracy: {:.2f}%'
          .format(epoch + 1, num_epochs, epoch_loss / len(train_loader), 100 * train_accuracy))




model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    test_epoch_loss = 0.0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_epoch_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    test_loss.append(test_epoch_loss / len(test_loader))
    test_acc.append(test_accuracy)
    print(("Test Accuracy: {:.2f}%").format(test_accuracy))

torch.save(model.state_dict(),"work1.pth")

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
target_loss_value = test_loss[0]
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