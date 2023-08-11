from optparse import Option
from turtle import forward
from matplotlib import transforms
from matplotlib.pyplot import title
import torch
import torch.nn.functional as F
from torch.nn import LSTM
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
from torch import no_grad, optim
import visdom


class Lstm(nn.Module):

    def __init__(self) -> None:
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(32 * 3, 128, batch_first=True, num_layers=5)
        self.line1 = nn.Linear(128, 128)
        self.line2 = nn.Linear(128, 100)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.line1(out[:, -1, :])
        out = self.line2(out)
        return out




# 全局变量设置
batch_size = 500
EPOCH = 5
LR = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, test_loader):
    total_correct = 0
    total_num = 0
    for step, (data, label) in enumerate(test_loader):
        data, label = data.to(device), label.to(device)
        data = data.view(-1, 32, 32 * 3)
        out = model(data)
        pred = out.argmax(dim=1)
        total_correct += torch.eq(pred, label).float().sum().item()
        total_num += data.size(dim=0)
        acc = total_correct / total_num
        return acc


#viz = visdom.Visdom()  # 训练可视化



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 图像归一化
])

model = Lstm().to(device)
criteon = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

    # 数据加载
trian_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(trian_set, batch_size=batch_size, shuffle=True)
test_set = datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

best_epoch, best_acc = 0, 0
global_step = 0
    #viz.line([0], [-1], win='loss', opts=dict(title="Loss"))
    #viz.line([0], [-1], win='Test_acc', opts=dict(title="Test_acc"))
    # 训练数据集
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 训练模型
total_step = len(train_loader)
for epoch in range(EPOCH):
    for i, (images, labels) in enumerate(train_loader):
        # 转换图像数据为序列形式
        images = images.view(-1, 32, 32 * 3).to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, EPOCH, i + 1, total_step,
                                                                     loss.item()))

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(-1, 32, 32 * 3).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('模型在 CIFAR-100 测试集上的准确率: {:.2f} %'.format(100 * correct / total))






