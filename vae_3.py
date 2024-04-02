import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from torchL.vae_2 import model, batch_size, z_dim, device, num_epochs, image_size, data_loader

# 加载模型
model.load_state_dict(torch.load('data/vae/vae' + str(num_epochs) + '.pth'))
model.eval()  # 设置为评估模式

# 生成图像
with torch.no_grad():
    # 随机生成的潜在向量
    z = torch.randn(batch_size, z_dim).to(device)
    out = model.decode(z).view(-1, 1, 28, 28)
    save_image(out, os.path.join('data/vae_samples/', 'sampled-' + str(num_epochs) + '.png'))
    print("Saved sampled images.")

    # 使用真实图像重构
    x, _ = next(iter(data_loader))
    x = x.to(device).view(-1, image_size)
    x_reconst, _, _ = model(x)
    x_concat = torch.cat([x.view(-1, 1, 28, 28), x_reconst.view(-1, 1, 28, 28)], dim=3)
    save_image(x_concat, os.path.join('data/vae_samples/', 'reconst-' + str(num_epochs) + '.png'))
    print("Saved reconstruction images.")

if __name__ == '__main__':
    reconsPath = 'data/vae_samples/sampled-' + str(num_epochs) + '.png'
    Image = mpimg.imread(reconsPath)
    plt.imshow(Image)  # 显示图像
    plt.axis('off')  # 不显示坐标轴
    plt.show()

    reconsPath2 = 'data/vae_samples/reconst-' + str(num_epochs) + '.png'
    Image2 = mpimg.imread(reconsPath2)
    plt.imshow(Image2)  # 显示图像
    plt.axis('off')  # 不显示坐标轴
    plt.show()
