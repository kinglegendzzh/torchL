import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# 设置参数
image_size = 784  # 28x28
h_dim = 400
z_dim = 20
num_epochs = 10
batch_size = 128
learning_rate = 0.001

# 下载MNIST训练集
dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def reparameterize(mu, log_var):
    std = torch.exp(log_var / 2)
    eps = torch.randn_like(std)
    return mu + eps * std


class VAE(nn.Module):
    def __init__(self, image_size=image_size, h_dim=h_dim, z_dim=z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)  # mu
        self.fc3 = nn.Linear(h_dim, z_dim)  # log_var
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, image_size))
        z = reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 初始化损失记录列表
epoch_losses = []
reconst_losses = []
kl_divs = []
if __name__ == '__main__':
    # 1. 设置训练循环
    # 训练过程遍历设定的训练周期数（num_epochs）。
    # 每一个周期（epoch）中，模型会处理数据加载器（data_loader）中的所有批次（batch）数据。
    for epoch in range(num_epochs):
        batch_losses = []
        batch_reconst_losses = []
        batch_kl_divs = []
        for i, (x, _) in enumerate(data_loader):
            # 2. 数据准备和模型前向传播
            # 对于每个批次的数据：
            # 将数据 x 转移到指定的设备（CPU或GPU）上，并调整其形状以适应模型输入
            # （例如，对于MNIST数据，将图像从2D形状 [batch_size, 1, 28, 28] 转换为1D形状 [batch_size, 784]）。
            # 执行模型的前向传播，计算重构的图像 x_reconst、潜在空间的均值 mu 和对数方差 log_var。
            x = x.to(device).view(-1, image_size)
            x_reconst, mu, log_var = model(x)

            # 3. 计算损失
            # 损失函数由两部分组成：重构损失和KL散度损失。
            # 重构损失 (reconst_loss)：使用二元交叉熵（Binary Cross Entropy, BCE）
            # 计算原始图像和重构图像之间的差异。reduction='sum' 表示对所有像素的损失求和。
            # KL散度损失 (kl_div)：计算潜在变量的先验分布（通常假设为标准正态分布）
            # 与后验分布（由模型通过 mu 和 log_var 定义）之间的KL散度。这一项鼓励潜在空间的分布接近先验分布。
            reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # 4. 反向传播和优化
            # 清零梯度缓存。
            # 计算总损失（重构损失 + KL散度损失），并进行反向传播。
            # 通过优化器（这里使用Adam）更新模型参数。
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            batch_reconst_losses.append(reconst_loss.item())
            batch_kl_divs.append(kl_div.item())

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch[{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Reconst Loss: {reconst_loss.item()}, KL Div: {kl_div.item()}")

        # 计算本epoch的平均损失
        epoch_loss = np.mean(batch_losses)
        epoch_reconst_loss = np.mean(batch_reconst_losses)
        epoch_kl_div = np.mean(batch_kl_divs)
        epoch_losses.append(epoch_loss)
        reconst_losses.append(epoch_reconst_loss)
        kl_divs.append(epoch_kl_div)

        print(
            f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Reconst Loss: {epoch_reconst_loss:.4f}, KL Div: {epoch_kl_div:.4f}")
    # 保存训练后的模型
    torch.save(model.state_dict(), 'data/vae/vae' + str(num_epochs) + '.pth')
    print('End')
    # 绘制损失图表
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label='Total Loss')
    plt.plot(reconst_losses, label='Reconstruction Loss')
    plt.plot(kl_divs, label='KL Divergence')
    plt.title('VAE Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
