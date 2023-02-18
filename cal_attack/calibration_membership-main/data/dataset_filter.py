from __future__ import print_function
import pandas as pd
import numpy as np
import os

import torch
from torch import nn



import argparse
import os




import torch.nn.functional as F


import torch
import torch.nn.parallel

from data.texas_undefended import Texas
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
import numpy as np
import tarfile
from sklearn.cluster import KMeans
from sklearn import datasets
import urllib
from data.purchase_undefended import PurchaseClassifier

torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES']='0'

np.set_printoptions(threshold=np.inf)
# 评估函数
def evaluate1(train_losses, out_losses, threshold, attack_base=None):
    true_positives= 0.0
    false_negatives= 0.0
    true_negatives= 0.0
    false_positives = 0.0
    if attack_base=='loss' or attack_base=='mean':
        for i in train_losses:
            if i <= threshold:
                true_positives += 1.0
            else:
                false_negatives += 1.0
        for i in out_losses:
            if i > threshold:
                true_negatives += 1.0
            else:
                false_positives += 1.0
    else:
        for i in train_losses:
            if i >= threshold:
                true_positives += 1.0
            else:
                false_negatives += 1.0
        for i in out_losses:
            if i < threshold:
                true_negatives += 1.0
            else:
                false_positives += 1.0

    fpr=false_positives / (false_positives + true_negatives)
    recall = true_positives / len(train_losses)
    precision = true_positives / (true_positives + false_positives)

    accuracy = (true_positives + true_negatives) / (len(train_losses)+len(out_losses))
    f1_score = (2.0*precision*recall)/(precision+recall)
    fnr = false_negatives / (true_positives +false_negatives)
    return fpr,fnr, precision, recall, accuracy,f1_score

def evaluate(train_losses, out_losses, threshold, attack_base=None):
    if attack_base=='loss' or attack_base=='mean':

        true_positives = (train_losses <= threshold).float()
        false_negatives= (train_losses > threshold).float()
        true_negatives = (out_losses > threshold).float()
        false_positives = (out_losses <= threshold).float()
    else:
        true_positives = (train_losses >= threshold).float()
        false_negatives = (train_losses < threshold).float()
        true_negatives = (out_losses < threshold).float()
        false_positives = (out_losses >= threshold).float()

    fpr=torch.sum(false_positives) / (torch.sum(false_positives) + torch.sum(true_negatives))
    recall = torch.sum(true_positives) / torch.sum(train_losses[:].float())
    precision = torch.sum(true_positives) / (torch.sum(true_positives) + torch.sum(false_positives))

    accuracy = (torch.sum(true_positives) + torch.sum(true_negatives)) / (torch.sum(out_losses[:].float()) + torch.sum(train_losses[:].float()))
    f1_score = (2.0*precision*recall)/(precision+recall)
    fnr = torch.sum(false_negatives) / (torch.sum(true_positives) + torch.sum(false_negatives))
    return fpr,fnr, precision, recall, accuracy,f1_score

'''提取原始的数据'''




def dataset_filter(dataset_name='location30',sheet_name = '0'):


    fpath = "./ratio_filter/" + dataset_name + "/ratio_filter.xlsx"

    df = pd.read_excel(fpath, sheet_name=sheet_name)
    index_list = df['index']
    # print(type(index_list))
    index_list = index_list.values
    index_list = np.array(list(filter(lambda x: not np.isnan(x), [x for x in index_list])))  # 去除nan
    index_list = index_list.astype(int)  # 浮点变int
    index_list = index_list.tolist()

    # print(index_list)
    # print(type(index_list))
    new_index_list = [j for i in index_list for j in range((i - 1) * 10, i*10)]

    if dataset_name == 'purchase100':
        shadow_index_list = [j for i in index_list for j in range(int((i - 1) * 10 + 197324*0.05), int(i * 10 + 197324*0.05))]
        new_test_index_list = [j for i in index_list for j in range(int((i - 1) * 10+197324*0.4), int(i*10+197324*0.4))]
    elif dataset_name == 'texas100':
        shadow_index_list = [j for i in index_list for j in range(int((i - 1) * 10 + 5000),int(i * 10 + 5000))]
        new_test_index_list = [j for i in index_list for j in range(int((i - 1) * 10 +  67330*0.3+10000), int(i * 10 +  67330*0.3+10000))]
    elif dataset_name == 'loction30':
        new_test_index_list = [j for i in index_list for j in
                               range(int((i - 1) * 10  + 1000), int(i * 10 + 1000))]
        shadow_index_list = new_index_list
    #目标模型训练集挑选后的结果
    X_target = X[new_index_list]
    Y_target = Y[new_index_list]

    #影子模型训练集挑选后的结果
    X_shadow = X[shadow_index_list]
    Y_shadow = Y[shadow_index_list]

    #测试集挑选后的结果
    X_test = X[new_test_index_list]
    Y_test = Y[new_test_index_list]
    # print(X_test)
    # print(Y_test)

    '''添加相关训练和攻击过程'''

    # 攻击需要的参数


    # 开始攻击
    if dataset_name == 'purchase100':
        # 获取攻击模型参数
        attacks_model = PurchaseClassifier()
        checkpoint = torch.load('./models/purchase_undefended_shadow/model_best.pth')
        attacks_model.load_state_dict(checkpoint['state_dict'], False)
        attacks_model = torch.nn.DataParallel(attacks_model).cuda()
        # 获取目标模型参数
        model = PurchaseClassifier()
        checkpoint = torch.load('./models/purchase_undefended/model_best.pth')
        model.load_state_dict(checkpoint['state_dict'], False)
        model = torch.nn.DataParallel(model).cuda()
        # 处理数据集
        train_size = len(X_target)
        out_size = len(X_test)
        # 计算成员经过校准的损失
        train_losses = np.zeros(train_size)
        for i in range(train_size):
            x = (X_target[i]).astype(np.float32)
            x = torch.tensor(x)
            # print(x)
            x = x.cuda()
            y = Y_target[i]
            y = torch.tensor(y)
            # print(y)
            y = y.cuda()
            output = model(x)[0]
            # print(output)
            loss = F.cross_entropy(output, y.long())
            attack_output = attacks_model(x)[0]
            attack_loss = F.cross_entropy(attack_output, y.long())
            train_losses[i] = loss - attack_loss
        # 计算非成员经过校准的损失
        out_losses = np.zeros(out_size)
        for i in range(out_size):
            x = (X_test[i]).astype(np.float32)
            x = torch.tensor(x)
            x = x.cuda()
            y = Y_test[i]
            y = torch.tensor(y)
            y = y.cuda()
            output = model(x)[0]
            loss = F.cross_entropy(output, y.long())
            attack_output = attacks_model(x)[0]
            attack_loss = F.cross_entropy(attack_output, y.long())
            out_losses[i] = loss - attack_loss
        # 计算度量值
        print(out_losses)
        fpr, fnr, precision, recall, accuracy, f1_score = evaluate1(train_losses,out_losses,0.0001,attack_base='loss')
        # 将度量值输出
        print(fpr, fnr, precision, recall, accuracy, f1_score)

        # 写入文件
        f = open('./purchase_loss/'+dataset_name+'_'+sheet_name+'_loss.txt','w')
        f.write("train_losses: "+str(train_losses)+"\n")
        f.write("out_losses:"+str(out_losses)+"\n")
        f.close()

        f = open('./purchase_result/'+dataset_name + '_' + sheet_name + 'result.txt', 'w')
        f.write("fpr: "+str(fpr)+" fnr: "+str(fnr)+" precision: "+str(precision)+" recall: "+str(recall)+" accuracy: "+str(accuracy)+" f1_score: "+str(f1_score)+"\n")
        f.close()
    elif dataset_name == 'texas100':
        # 获取攻击模型参数
        attacks_model = Texas()
        checkpoint = torch.load('./models/texas_undefended_shadow/model_best.pth')
        attacks_model.load_state_dict(checkpoint['state_dict'], False)
        attacks_model = torch.nn.DataParallel(attacks_model).cuda()
        # 获取目标模型参数
        model = Texas()
        checkpoint = torch.load('./models/texas_undefended/model_best.pth')
        model.load_state_dict(checkpoint['state_dict'], False)
        model = torch.nn.DataParallel(model).cuda()
        # 处理数据集
        train_size = len(X_target)
        out_size = len(X_test)
        # 计算成员经过校准的损失
        train_losses = np.zeros(train_size)
        for i in range(train_size):
            x = (X_target[i]).astype(np.float32)
            x = torch.tensor(x)
            print("x:.....:",x)
            x = x.cuda()
            y = Y_target[i]
            y = torch.tensor(y)
            # print(y)
            y = y.cuda()
            output = model(x)[0]
            print("output:",output)
            loss = F.cross_entropy(output, y.long())
            attack_output = attacks_model(x)[0]
            attack_loss = F.cross_entropy(attack_output, y.long())
            train_losses[i] = loss - attack_loss
        # 计算非成员经过校准的损失
        out_losses = np.zeros(out_size)
        for i in range(out_size):
            x = (X_test[i]).astype(np.float32)
            x = torch.tensor(x)
            x = x.cuda()
            y = Y_test[i]
            y = torch.tensor(y)
            y = y.cuda()
            output = model(x)[0]
            loss = F.cross_entropy(output, y.long())
            attack_output = attacks_model(x)[0]
            attack_loss = F.cross_entropy(attack_output, y.long())
            out_losses[i] = loss - attack_loss
        # 计算度量值
        print(out_losses)
        fpr, fnr, precision, recall, accuracy, f1_score = evaluate1(train_losses, out_losses, 0.0001,
                                                                    attack_base='loss')
        # 将度量值输出
        print(fpr, fnr, precision, recall, accuracy, f1_score)
        # 写入文件
        f = open('./texas_loss/'+dataset_name + '_' + sheet_name + '_loss.txt', 'w')
        f.write("train_losses: " + str(train_losses) + "\n")
        f.write("out_losses:" + str(out_losses) + "\n")
        f.close()

        f = open('./texas_result/'+dataset_name + '_' + sheet_name + 'result.txt', 'w')
        f.write("fpr: " + str(fpr) + " fnr: " + str(fnr) + " precision: " + str(precision) + " recall: " + str(
            recall) + " accuracy: " + str(accuracy) + " f1_score: " + str(f1_score) + "\n")
        f.close()



dataset_names = ['laocation30']
#dataset_names = ['texas100']
for dataset_name in dataset_names:
    if dataset_name == 'location30':
        path = '.\datasets\location\shuffle_index.npz'
        shuffle_index_list = np.load(path, allow_pickle=True)['x']
        path = '.\datasets\location\data_complete.npz'
        dataset = np.load(path, allow_pickle=True)
        X = dataset['x'][shuffle_index_list]
        Y = dataset['y'][shuffle_index_list]
    elif(dataset_name == 'purchase100'):
        path = './dataset_shuffle/random_r_purchase100.npy'
        shuffle_index_list = np.load(path, allow_pickle=True)
        path = './datasets/purchase/dataset_purchase'
        data_set = np.genfromtxt(path, delimiter=',')
        X = data_set[:, 1:].astype(np.float64)
        Y = (data_set[:, 0]).astype(np.int32) - 1
        X = X[shuffle_index_list]
        Y = Y[shuffle_index_list]

    elif(dataset_name == 'texas100'):
        path = './dataset_shuffle/random_r_texas100.npy'
        shuffle_index_list = np.load(path, allow_pickle=True)
        DATASET_PATH = './datasets/texas/'
        DATASET_FEATURES = os.path.join(DATASET_PATH, 'texas/100/feats')
        DATASET_LABELS = os.path.join(DATASET_PATH, 'texas/100/labels')
        data_set_features = np.genfromtxt(DATASET_FEATURES, delimiter=',')
        data_set_label = np.genfromtxt(DATASET_LABELS, delimiter=',')

        X = data_set_features.astype(np.float64)
        Y = data_set_label.astype(np.int32) - 1
        X = X[shuffle_index_list]
        Y = Y[shuffle_index_list]

    print(f'{dataset_name=}')
    fpath = "./ratio_filter/" + dataset_name + "/ratio_filter.xlsx"
    reader = pd.ExcelFile(fpath)
    sheet_names = reader.sheet_names
    for sheet_name in sheet_names:
        print(f'{sheet_name=}')
        dataset_filter(dataset_name=dataset_name,sheet_name = sheet_name)