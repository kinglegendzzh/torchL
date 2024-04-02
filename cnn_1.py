# 导入库及下载数据
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# print('loader work done!')

# 随机查看部分数据
from matplotlib import pyplot as plt
import numpy as np
import torchvision


# %matplotlib inline
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    # print('showing img')


if __name__ == '__main__':
    # 这是在 Windows 和 macOS 上使用 spawn 或 forkserver 时激活多进程支持的标准方式。
    # 在 Unix 上，它有助于防止在产生新进程时运行代码
    # 显示图像
    # 随机获取部分训练数据
    dataiter = iter(trainloader)
    # print(dataiter.__sizeof__())
    images, labels = next(dataiter)
    # 显示图像
    imshow(torchvision.utils.make_grid(images))
    # 打印标签
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
