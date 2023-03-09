import torch
from MMD import mmd_rbf
from torch.autograd import Variable
import random
import math


a = [[random.random() for i in range(2)]for j in range(2)]
b = [[random.random() for i in range(2)]for j in range(2)]
#e = [[random.random() for i in range(2)]for j in range(2)]
c = []
for i in range(2):
    temp = []
    for j in range(2):
        temp.append(a[i][j]+1)
    c.append(temp)
d = []
for i in range(2):
    temp = []
    for j in range(2):
        temp.append(b[i][j]+1)
    d.append(temp)

e = []
for i in range(2):
    temp = []
    for j in range(2):
        temp.append(a[i][j]+2)
    e.append(temp)
f = []
for i in range(2):
    temp = []
    for j in range(2):
        temp.append(b[i][j]+2)
    f.append(temp)


# c = [[random.random() for i in range(2)]for j in range(2)]
# d = [[random.random() for i in range(2)]for j in range(2)]
a = torch.tensor(a)
b = torch.tensor(b)
c = torch.tensor(c)
d = torch.tensor(d)
e = torch.tensor(e)
f = torch.tensor(f)
a,b,c,d = Variable(a), Variable(b),Variable(c), Variable(d)

print((math.pow(mmd_rbf(a,b),0.5)+math.pow(mmd_rbf(c,d),0.5))/2)
print('mmd a,b :{}'.format(mmd_rbf(a,b)))
print('mmd c,d :{}'.format(mmd_rbf(c,d)))
print('mmd e,f :{}'.format(mmd_rbf(e,f)))
# print('a:{}'.format((a)))
ac = torch.cat((a,c),dim = 0)
# print('a+c:{}'.format((ac)))
bd = torch.cat((b,d),dim = 0)
ae = torch.cat((a,e),dim = 0)
bf = torch.cat((b,f),dim = 0)
print('mmd ac,bd:{}'.format(mmd_rbf(ac,bd)))
print('mmd ae,bf:{}'.format(mmd_rbf(ae,bf)))
# print(math.pow(mmd_rbf(ac,bd),0.5))
# mmd_rbf(a,b)
# mmd_rbf(c,d)


# a = [[11,5,5,9],[5,56,6,4],[5,3,98,5]]
# a = torch.tensor((a))
# print(a[:2,:2])
# print(a[2:,2:])
# print(a[:-2,:-2])
# print(a[-2:,-2:])

# a = [11,9,5,4]
# a = torch.tensor(a)
# print(a[1:3])

# b = [11,5,5,9]
# b = torch.tensor(b)
# print(b[-1:-3])
# print(b[3:4])
# print(b[:-1])
# print(b[0:3])
#
# a = [11,9,5,4]
# a = torch.tensor(a)
#
# b = [11,5,5,9]
# b = torch.tensor(b)
#
# print((a-b)**2)

# a = [torch.rand(4) for i in range(3)]
# b = [torch.rand(4) for i in range(3)]
# print((a-b)**2)

