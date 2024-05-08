# 训练模型
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_1 import trainloader
from cnn_2 import device, net

if __name__ == '__main__':
    # 选择优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取训练数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 权重参数梯度清零
            optimizer.zero_grad()

            # 正向及反向传播
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 显示损失值
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
