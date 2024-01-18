import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sam import SAM
import torchvision
from Esam import ESAM
from KSAM import K_SAM

BATCH_SIZE = 512  # 批次大小
EPOCHS = 20  # 总共训练批次
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # 28x28
        # self.pool1 = nn.MaxPool2d(2, 2)  # 14x14
        # self.conv2 = nn.Conv2d(6, 16, 5)  # 10x10
        # self.pool2 = nn.MaxPool2d(2, 2)  # 5x5
        # self.conv3 = nn.Conv2d(16, 120, 5)
        # self.fc1 = nn.Linear(120, 84)
        # self.fc2 = nn.Linear(84, 10)
        self.conv1 = nn.Conv2d(3, 6, 5, padding=0)  # 28x28
        self.pool1 = nn.MaxPool2d(2, 2)  # 14x14
        self.conv2 = nn.Conv2d(6, 16, 5)  # 10x10
        self.pool2 = nn.MaxPool2d(2, 2)  # 5x5
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
    def forward(self, x):
        #relu激活函数
        in_size = x.size(0)
        out = self.conv1(x)  # 24
        out = F.relu(out)
        out = self.pool1(out)  # 12
        out = self.conv2(out)  # 10
        out = F.relu(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out


# def train(model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if (batch_idx + 1) % 30 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss.item()))


# def train(model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         # 选择top-K样本
#         topk1, _ = torch.topk(loss, k=optimizer.defaults['K1'])
#         topk2, _ = torch.topk(loss, k=optimizer.defaults['K2'])
#         # 反向传播
#         optimizer.zero_grad()
#         output = model(data[topk1])
#         loss = F.nll_loss(output, target[topk1])
#         loss.backward()
#         optimizer.first_step(zero_grad=True)
#         # 第二次反向传播
#         output = model(data[topk2])
#         loss = F.nll_loss(output, target[topk2])
#         loss.backward()
#         optimizer.second_step(zero_grad=False)
#         if (batch_idx + 1) % 30 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss.item()))



def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # 第二次前向传播
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.second_step(zero_grad=False)

        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))





def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=BATCH_SIZE, shuffle=True)
    #
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('data', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])),
    #     batch_size=BATCH_SIZE, shuffle=True)
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                               shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                              shuffle=True)

    model = ConvNet().to(DEVICE)

    #SAM优化方法   SGD
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9, rho=0.01)
    # base_optimizer = torch.optim.SGD
    # optimizer = K_SAM(model.parameters(), base_optimizer, lr=0.001, momentum=0.9, rho=0.01, K1=5, K2=10)

    # base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # optimizer = ESAM(model.parameters(), base_optimizer, rho=0.01)

    # SAM优化方法    Adam
    # base_optimizer = torch.optim.Adam
    # optimizer = SAM(model.parameters(), base_optimizer, lr=0.01, rho=0.1)

    #SGD优化方法
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    #Adam优化方法
    # optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=0, amsgrad=False)

    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test(model, DEVICE, test_loader)
