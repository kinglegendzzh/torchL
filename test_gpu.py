import torch

if __name__ == '__main__':
    #test cuda
    print("Support CUDA ?: ", torch.cuda.is_available())
    x = torch.tensor([10.0])
    x = x.cuda()
    print(x)
