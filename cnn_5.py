# 像Keras一样显示各层参数
# 用Keras显示一个模型参数及其结构非常方便，结果详细且规整。
# 当然，PyTorch也可 以显示模型参数，但结果不是很理想。
# 这里介绍一种显示各层参数的方法，其结果类似 Keras的展示结果。
import collections
import torch
import torch.nn as nn
from torch import tensor

from torchL.cnn_2 import CNNNet


# 先定义汇总各层网络参数的函数
def paras_summary(input_size, model):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)
            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = collections.OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            summary[m_key]['output_shape'] = list(output.size())
            summary[m_key]['output_shape'][0] = -1
            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                if module.weight.requires_grad:
                    summary[m_key]['trainable'] = True
                else:
                    summary[m_key]['trainable'] = False
            if hasattr(module, 'bias'):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if not isinstance(module, nn.Sequential) and \
                not isinstance(module, nn.ModuleList) and \
                not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [torch.rand(1, *in_size) for in_size in input_size]
    else:
        x = torch.rand(1, *input_size)
    # create properties
    summary = collections.OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()
    return summary


# 确定输入及实例化模型

net = CNNNet()  # 输入格式为[c,h,w]即通道数，图像的高级宽度
input_size = [3, 32, 32]
paras_summary(input_size, net)
collections.OrderedDict([('Conv2d-1',
                          collections.OrderedDict([('input_shape', [-1, 3, 32, 32]),
                                                   ('output_shape', [-1, 16, 28, 28]),

                                                   ('trainable', True),
                                                   ('nb_params', tensor(1216))])),
                         ('MaxPool2d-2',
                          collections.OrderedDict([('input_shape', [-1, 16, 28, 28]),
                                                   ('output_shape', [-1, 16, 14, 14]),
                                                   ('nb_params', 0)])),
                         ('Conv2d-3',
                          collections.OrderedDict([('input_shape', [-1, 16, 14, 14]),
                                                   ('output_shape', [-1, 36, 12, 12]),
                                                   ('trainable', True),
                                                   ('nb_params', tensor(5220))])),
                         ('MaxPool2d-4',
                          collections.OrderedDict([('input_shape', [-1, 36, 12, 12]),
                                                   ('output_shape', [-1, 36, 6, 6]),
                                                   ('nb_params', 0)])),
                         ('Linear-5',
                          collections.OrderedDict([('input_shape', [-1, 1296]),
                                                   ('output_shape', [-1, 128]),
                                                   ('trainable', True),
                                                   ('nb_params', tensor(166016))])),
                         ('Linear-6',
                          collections.OrderedDict([('input_shape', [-1, 128]),
                                                   ('output_shape', [-1, 10]),
                                                   ('trainable', True),
                                                   ('nb_params', tensor(1290))]))])
