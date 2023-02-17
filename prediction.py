import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *

def prediction(dataloader,device,target_net,batch_size):
    prediction_npy =[]
    label_npy =[]

    #index
    index = 0
    for batch_idx, (inputs, label) in enumerate(dataloader):
        index += 1
        inputs, label = inputs.to(device), label.to(device)
        tr_prediction = target_net(inputs)
        tr_prediction = tr_prediction.cpu().detach().numpy()
        tr_label = label.cpu().detach().numpy()
        for j in range((batch_size)):
            prediction_npy.append(tr_prediction[j].tolist())
            label_npy.append(tr_label[j].tolist())
        if index == 10:
            break
    return prediction_npy,label_npy

'''
input: weight of model, source data
output: confidence score
'''
#target_model_path,shadow_model_path,target_data_path,shadow_data_path,model,data
#saved_path './saved_predictions/cifar_model_regularization/prediction_result/'
def data_process(target_model_path,shadow_model_path,model,data,saved_path,target_data_path='',shadow_data_path=''):
    #
    # path1 = './regularization_array/lamda = {}targets/'.format(lamda)
    # if not os.path.exists(path1):
    #     os.makedirs(path1)
    # path2 = './regularization_array/lamda = {}shadow/'.format(lamda)
    # if not os.path.exists(path2):
    #     os.makedirs(path2)


#model
    if(model=='SimpleDLA'):
        target_net = SimpleDLA().cuda()
        target_net = nn.DataParallel(target_net)
        target_net.load_state_dict(torch.load(target_model_path)['net'])

        shadow_net = SimpleDLA().cuda()
        shadow_net = nn.DataParallel(shadow_net)
        shadow_net.load_state_dict(torch.load(shadow_model_path)['net'])
    #elif(model == ''):


    #data
    print('==> Preparing data..')
    if(data =='cifar10'):
        batch_size = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)


        #target_trainset,shadow_trainset,shadow_testset = torch.utils.data.random_split(trainset, [20000,20000, 10000], generator=torch.Generator().manual_seed(42))
        target_trainset, shadow_trainset, shadow_testset = torch.utils.data.random_split(trainset, [20000, 20000, 10000],
                                                                                         generator=torch.Generator())

        target_trainloader = torch.utils.data.DataLoader(
            target_trainset, batch_size=100, shuffle=True, num_workers=2)

        shadow_trainloader = torch.utils.data.DataLoader(
            shadow_trainset, batch_size=100, shuffle=True, num_workers=2)

        shadow_testloader = torch.utils.data.DataLoader(
            shadow_testset, batch_size=100, shuffle=True, num_workers=2)

        # testset = torchvision.datasets.MNIST(
        #     root='./data', train=False, download=True, transform=transform_test)

        target_testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        target_testloader = torch.utils.data.DataLoader(
            target_testset, batch_size=100, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

    if(torch.cuda.is_available()):
        device = 'cuda'
    else:
        device = 'cpu'
    print(device)
    #prediction

    if not(os.path.exists(saved_path+'/'+data)):
        os.mkdir(saved_path+'/'+data)
    saved_path = saved_path+'/'+data +'/'

    #target
    tr_prediction_npy,tr_label_npy = prediction(target_trainloader, device, target_net,batch_size)
    te_prediction_npy, te_label_npy = prediction(target_testloader, device, target_net,batch_size)

    np.save(saved_path + 'tr_target', np.array(tr_prediction_npy))
    np.save(saved_path + 'tr_target_label', np.array(tr_label_npy))
    np.save(saved_path + 'te_target', np.array(te_prediction_npy))
    np.save(saved_path + 'te_target_label', np.array(te_label_npy))

    #shadow
    tr_prediction_npy, tr_label_npy = prediction(shadow_trainloader, device, target_net,batch_size)
    te_prediction_npy, te_label_npy = prediction(shadow_testloader, device, target_net,batch_size)

    np.save(saved_path + 'tr_shadow', np.array(tr_prediction_npy))
    np.save(saved_path + 'tr_shadow_label', np.array(tr_label_npy))
    np.save(saved_path + 'te_shadow', np.array(te_prediction_npy))
    np.save(saved_path + 'te_shadow_label', np.array(te_label_npy))

import torch
import numpy
import random

if __name__ == '__main__':

    seed = 42
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed() GPU
    torch.manual_seed(seed)
    # CPU

    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)

    #parameter
    import argparse
    # # 1、ArgumentParser()
    # #
    # # 2、 add_argument()
    # #
    # # 3、parse_args()
    parser = argparse.ArgumentParser()


    parser.add_argument('-m',"--model", help="h", type=str,default='SimpleDLA')
    parser.add_argument('-d',"--data", help="h", type=str,default='cifar10')

    parser.add_argument('-tmp',"--target_model_path", help="target_model_path", type=str,default='./pretrained_models/cifar10_target_model.pth')
    parser.add_argument('-smp',"--shadow_model_path", help="shadow_model_path", type=str,default='./pretrained_models/cifar10_shadow_model.pth')
    parser.add_argument('-tdp',"--target_data_path", help="h", type=str,default='')
    parser.add_argument('-sdp',"--shadow_data_path", help="h", type=str,default='')
    parser.add_argument('-sp',"--saved_path",help="h", type=str,default='./saved_predictions/')


    args = parser.parse_args()

    from prediction import data_process

    data_process(args.target_model_path,args.shadow_model_path,args.model,args.data,args.saved_path)