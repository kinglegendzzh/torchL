import torch

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())

print(f'已用GPU内存大小{torch.cuda.memory_allocated()}')
print(f'清除缓存{torch.cuda.empty_cache()}')