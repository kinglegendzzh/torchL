# 使用现代经典模型提升性能
# 前面通过使用一些比较简单的模型对数据集CIFAR-10进行分类，精度在68%左右，
# 然后使用模型集成的方法，同样是这些模型，但精度却提升到74%左右。虽有一定提升， 但结果还是不够理想。
# 精度不够很大程度与模型有关，前面我们介绍的一些现代经典网络，
# 在大赛中都取得 了不俗的成绩，说明其模型结构有很多突出的优点，
# 所以，人们经常直接使用这些经典模 型作为数据的分类器。这里我们就用VGG16这个模型，
# 来对数据集IFAR10进行分类，直 接效果非常不错，精度一下子就提高到90%左右，效果非常显著。
from torch import nn

cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


VGG16 = VGG('VGG16')
