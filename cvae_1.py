import time

import torch
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# 参数设置
image_size = 784  # 图像尺寸 28x28
h_dim = 400
z_dim = 20
c_dim = 10  # 类别数目（0-9）
num_epochs = 60
batch_size = 128 * 350
learning_rate = 0.0211

# 数据集加载
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
# data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


# 条件变分自编码器 (CVAE) 定义
class CVAE(torch.nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20, c_dim=10):
        super(CVAE, self).__init__()
        self.fc1 = torch.nn.Linear(image_size + c_dim, h_dim)
        self.fc2 = torch.nn.Linear(h_dim, z_dim)  # mu
        self.fc3 = torch.nn.Linear(h_dim, z_dim)  # log var
        self.fc4 = torch.nn.Linear(z_dim + c_dim, h_dim)
        self.fc5 = torch.nn.Linear(h_dim, image_size)

    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        h = F.relu(self.fc1(inputs))
        return self.fc2(h), self.fc3(h)

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h = F.relu(self.fc4(inputs))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x, c):
        mu, log_var = self.encode(x.view(-1, image_size), c)
        std = log_var.mul(0.5).exp_()
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        x_reconst = self.decode(z, c)
        return x_reconst, mu, log_var


def one_hot(labels, dimension=10):
    targets = torch.zeros(labels.size(0), dimension)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(labels.device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = CVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
epoch_losses = []
if __name__ == '__main__':
    start_time = time.time()
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for idx, (x, labels) in enumerate(data_loader):
            x = x.to(device).view(-1, image_size)
            labels = one_hot(labels).to(device)

            optimizer.zero_grad()
            x_reconst, mu, log_var = model(x, labels)

            reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            loss = reconst_loss + kl_div
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (idx + 1) % 100 == 0:
                print(
                    f"Epoch[{epoch + 1}/{num_epochs}], Step [{idx + 1}/{len(data_loader)}], Reconst Loss: {reconst_loss.item()}, KL Div: {kl_div.item()}")
        scheduler.step()

        avg_loss = train_loss / len(data_loader.dataset)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    end_time = time.time()  # 训练结束后的时间
    training_time = end_time - start_time  # 计算总训练时间
    print(f"Total training time: {training_time:.2f} seconds.")
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_losses, label='Train Loss', marker='o')
    plt.title('Training Loss Over Epochs：(cvae' + str(num_epochs) + str(batch_size) + str(learning_rate) +'.pth)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 保存模型
    if not os.path.exists('cvae_models'):
        os.makedirs('cvae_models')
    torch.save(model.state_dict(), 'cvae_models/cvae' + str(num_epochs) + str(batch_size) + str(learning_rate) +'.pth')
    print("Model saved.")
    torch.cuda.empty_cache()
