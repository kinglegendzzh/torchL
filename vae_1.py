import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from matplotlib import image as mpimg
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 30
batch_size = 128
learning_rate = 0.001

# 下载MNIST训练集，这里因已下载，故download=False
# 如果需要下载，设置download=True将自动下载
dataset = torchvision.datasets.MNIST(root='data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=False)
# 数据加载
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


# 定义AVE模型
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    # 用mu，log_var生成一个潜在空间点z，mu，log_var为两个统计参数，我们假设 #这个假设分布能生成图像。
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


# 设置PyTorch在哪块GPU上运行，这里假设使用序号为1的这块GPU.
# torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
x, _ = next(iter(data_loader))  # 从数据加载器获取一批数据
x = x.to(device).view(-1, image_size)  # 调整形状以匹配模型输入，如果需要的话
sample_dir = 'data/vae_samples/'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
with torch.no_grad():
    # 保存采样图像，即潜在向量Z通过解码器生成的新图像
    z = torch.randn(batch_size, z_dim).to(device)
    out = model.decode(z).view(-1, 1, 28, 28)
    save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(num_epochs)))
    # 保存重构图像，即原图像通过解码器生成的图像
    out, _, _ = model(x)
    x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
    save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(num_epochs)))

reconsPath = 'data/vae_samples/reconst-30.png'
Image = mpimg.imread(reconsPath)
plt.imshow(Image)  # 显示图像
plt.axis('off')  # 不显示坐标轴
plt.show()
