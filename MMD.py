# -*- coding: gbk -*-
#MMD
import torch
from torch.autograd import Variable

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    Transform the source domain data and target domain data into a kernel matrix
    Params:
     source:
     target:
     kernel_mul:
     kernel_num:
     fix_sigma:fix kernal parameter sigma and use sample kernel
 Return:
  sum(kernel_val):
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])
    # #print('n_samples: {}'.format(n_samples))

    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #print(f'{total=}')
    # create n+m files all of them are same as total
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #print(f'{total.size()=}')
    #print(f'{total.size(0)=}')
    #print(f'{total0=}')

    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    L2_distance_square = ((total0-total1)**2).sum(2)
    #print(f'{((total0-total1)**2).sum()=}')
    #print(f'{L2_distance_square=}')

    if fix_sigma:
        ##print(fix_sigma)
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance_square) / (n_samples**2-n_samples)
        #print(f'{bandwidth=}')
        #choose proper sigma
        ##print('bandwidth: {}'.format(bandwidth))

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    ##print(bandwidth_list)

    #mathma
    kernel_val = [torch.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    # #print(sum(kernel_val))
    #shape: n*n
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    (Radial Basis Function Kernel)
 Return:
  loss: MMD loss
    '''
    source_num = int(source.size()[0])#
    target_num = int(target.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = torch.mean(kernels[:source_num, :source_num])
    YY = torch.mean(kernels[source_num:, source_num:])
    XY = torch.mean(kernels[:source_num, source_num:])
    YX = torch.mean(kernels[source_num:, :source_num])
    loss = XX + YY -XY - YX
    return loss#

def mmd_understanding():
    '''
        analysis mmd，
    '''
    import random
    import matplotlib
    import matplotlib.pyplot as plt

    SAMPLE_SIZE = 500
    buckets = 50


    plt.subplot(1,3,1)
    plt.xlabel("random.normalvariate")
    mu = 0.07
    sigma = 0.15#
    res1 = [random.normalvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)]
    plt.hist(res1, buckets)

    #beta
    plt.subplot(1,3,2)
    plt.xlabel("random.betavariate")
    alpha = 1
    beta = 10
    res2 = [random.betavariate(alpha, beta) for _ in range(1, SAMPLE_SIZE)]
    plt.hist(res2, buckets
             )
    #
    plt.subplot(1, 3, 3)
    plt.xlabel("random.lognormalvariate")
    mu2 = -1.2
    sigma2 = 0.4  #
    res3 = [random.lognormvariate(mu2, sigma2) for _ in range(1, SAMPLE_SIZE)]
    plt.hist(res3, buckets)

    plt.savefig('data.jpg')
    plt.show()
    from torch.autograd import Variable


    diff_1 = []
    for i in range(10):
        diff_1.append([random.normalvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)])

    diff_2 = []
    for i in range(10):
        diff_2.append([random.betavariate(alpha, beta) for _ in range(1, SAMPLE_SIZE)])

    #print('diff_1:{}'.format(diff_1))
    X = torch.Tensor(diff_1)
    Y = torch.Tensor(diff_2)
    X,Y = Variable(X), Variable(Y)

    print(mmd_rbf(X,Y))
    # #print('X,Y:')
    # #print(X,Y)
    from torch.autograd import Variable


    same_1 = []
    for i in range(10):
        same_1.append([random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)])

    same_2 = []
    for i in range(10):
        same_2.append([random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)])

    X = torch.Tensor(same_1)
    Y = torch.Tensor(same_2)
    X,Y = Variable(X), Variable(Y)
    print(mmd_rbf(X,Y))


    diff_1 = []
    for i in range(10):
        diff_1.append([random.normalvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)])

    diff_2 = []
    for i in range(10):
        diff_2.append([random.lognormvariate(mu2, sigma2) for _ in range(1, SAMPLE_SIZE)])

    # print('diff_1:{}'.format(diff_1))
    X = torch.Tensor(diff_1)
    Y = torch.Tensor(diff_2)
    X, Y = Variable(X), Variable(Y)

    print(mmd_rbf(X, Y))

