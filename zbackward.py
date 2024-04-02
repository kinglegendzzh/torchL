import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#定义输入张量x
x=torch.Tensor([2])
#初始化权重参数W,偏移量b、并设置require_grad属性为True，为自动求导
w=torch.randn(1,requires_grad=True)
b=torch.randn(1,requires_grad=True)
#实现前向传播
y=torch.mul(w,x) #等价于w*x
z=torch.add(y,b) #等价于y+b
#查看x,w，b页子节点的requite_grad属性
print("x,w,b的require_grad属性分别为:{},{},{}".format(x.requires_grad,w.requires_grad,b.requires_grad))

#查看非叶子节点的requres_grad属性,
print("y，z的requires_grad属性分别为:{},{}".format(y.requires_grad,z.requires_grad))
#因与w，b有依赖关系，故y，z的requires_grad属性也是:True,True
#查看各节点是否为叶子节点
print("x，w，b，y，z的是否为叶子节点:{},{},{},{},{}".format(x.is_leaf,w.is_leaf,b.is_leaf,y.is_leaf,z.is_leaf))
#x，w，b，y，z的是否为叶子节点:True,True,True,False,False
#查看叶子节点的grad_fn属性
print("x，w，b的grad_fn属性:{},{},{}".format(x.grad_fn,w.grad_fn,b.grad_fn))
#因x，w，b为用户创建的，为通过其他张量计算得到，故x，w，b的grad_fn属性:None,None,None
#查看非叶子节点的grad_fn属性
print("y，z的是否为叶子节点:{},{}".format(y.grad_fn,z.grad_fn))
#y，z的是否为叶子节点:<MulBackward0 object at 0x7f923e85dda0>,<AddBackward0 object at 0x7f923e85d9b0>

#基于z张量进行梯度反向传播,执行backward之后计算图会自动清空，
z.backward() #如果需要多次使用backward，需要修改参数retain_graph为True，此时梯度是累加的
# z.backward(retain_graph=True)
#查看叶子节点的梯度，x是叶子节点但它无须求导，故其梯度为None
print("参数w,b的梯度分别为:{},{},{}".format(w.grad,b.grad,x.grad))
#参数w,b的梯度分别为:tensor([2.]),tensor([1.]),None
#非叶子节点的梯度，执行backward之后，会自动清空
print("非叶子节点y,z的梯度分别为:{},{}".format(y.grad,z.grad))
#非叶子节点y,z的梯度分别为:None,None