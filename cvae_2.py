import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from torchL.cvae_1 import model, batch_size, z_dim, device, num_epochs, image_size, data_loader, learning_rate


# 生成指定数字的图像
def generate_digit(model, digit, device='cpu'):
    model.eval()
    with torch.no_grad():
        c = torch.zeros(1, 10).to(device)
        c[0, digit] = 1
        z = torch.randn(1, z_dim).to(device)
        sample = model.decode(z, c).view(28, 28).cpu().numpy()
        plt.imshow(sample, cmap='gray')
        plt.show()


if __name__ == '__main__':
    # 加载模型并生成数字
    model.load_state_dict(torch.load('cvae_models/cvae' + str(num_epochs) + str(batch_size) + str(learning_rate) + '.pth'))
    breaking = True
    while breaking:
        inn = input("请输入你想要生成的数字")
        if inn == '/':
            break
        if 0 <= int(inn) <= 9:
            generate_digit(model, int(inn), device)
    print('End')
