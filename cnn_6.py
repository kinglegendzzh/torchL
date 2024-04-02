# 导入nn及优化器
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from torchL.cnn_1 import trainloader
from torchL.cnn_2 import device


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

EPOCHES = 2
LR = 0.001
mlps=[net1.to(device),net2.to(device),net3.to(device)]
optimizer=torch.optim.Adam([{"params":mlp.parameters()} for mlp in mlps],lr=LR)
loss_function=nn.CrossEntropyLoss()
for ep in range(EPOCHES):
    for img,label in trainloader:
        img,label=img.to(device),label.to(device) optimizer.zero_grad()#10个网络清除梯度
for mlp in mlps:
mlp.train()
out=mlp(img) loss=loss_function(out,label) loss.backward()#网络获得的梯度
           optimizer.step()
       pre=[]
       vote_correct=0
       mlps_correct=[0 for i in range(len(mlps))]
       for img,label in testloader:
           img,label=img.to(device),label.to(device)
           for i, mlp in  enumerate( mlps):
               mlp.eval()
               out=mlp(img)
_,prediction=torch.max(out,1) #按行取最大值 pre_num=prediction.cpu().numpy() mlps_correct[i]+=(pre_num==label.cpu().numpy()).sum()
               pre.append(pre_num)
           arr=np.array(pre)
           pre.clear()
           result=[Counter(arr[:,i]).most_common(1)[0][0] for i in range(BATCHSIZE)]
           vote_correct+=(result == label.cpu().numpy()).sum()
print("epoch:" + str(ep)+"集成模型的正确率"+str(vote_correct/len(testloader)))
for idx, coreect in enumerate( mlps_correct): print("模型"+str(idx)+"的正确率为:"+str(coreect/len(testloader)))