#choose data by risk_score
import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import math
import sys
import urllib
import pickle
import argparse
#from utils import *
#sys.path.append('../')
from membership_inference_attacks import black_box_benchmarks
import privacy_risk_score_utils
import pandas as pd


#ratio_filter: delete data by ratio and choose ratio by reference risk_score
def ratio_filter_analysis(ratio_list,shadow_train_performance, shadow_test_performance,
                                       target_train_performance, target_test_performance,risk_score,num_classes):
    origin_length = len(target_train_performance[0])
    # print('origin_length:')
    # print(origin_length)
    return_ratio = []
    index = 0 #

    MIA = black_box_benchmarks(shadow_train_performance, shadow_test_performance,
                               target_train_performance, target_test_performance, num_classes=num_classes,
                               length=origin_length)
    TP, FP, FN, TN, R, P, F1, FPR, MA,acc = MIA._mem_inf_benchmarks()
    print('filter ratio = {}'.format(0))
    print('TP:{TP}, FP:{FP}, FN:{FN},TN={TN}'.format(TP=TP, FP=FP, FN=FN, TN=TN))
    print('R:{R}, P:{P}, F1:{F1},FPR={FPR},MA={MA},acc={acc}'.format(R=R, P=P, F1=F1, FPR=FPR, MA=MA,acc =acc))

    for i in range(1000):
        delete_list = []
        thre = i/1000 #step of risk_score threshold
        #print('thre:{}'.format(thre))
        for j in range(origin_length):
            if risk_score[j] < thre:
                #print('score:{}'.format(risk_score[j]))
                delete_list.append(j)
        #print(1000-len(delete_list))
        #print('*')
        #print(target_train_performance[0])
        target_train_performance_filtered = (np.delete(target_train_performance[0],delete_list,0),np.delete(target_train_performance[1],delete_list,0))
        #target_test_performance_filtered = (np.delete(target_test_performance[0],delete_list,0),np.delete(target_test_performance[1],delete_list,0))
        #print(target_train_performance_filtered[0])
        data_length = len(target_train_performance_filtered[0])


        if (1-(data_length+origin_length)/(2*origin_length))>=ratio_list[index]:
            index+=1
            #print(1-((data_length)/origin_length))
            return_ratio.append(1-(data_length+origin_length)/(2*origin_length))
            MIA = black_box_benchmarks(shadow_train_performance, shadow_test_performance,
                                       target_train_performance_filtered, target_test_performance, num_classes=num_classes,length = data_length)
            TP, FP, FN, TN,R, P, F1,FPR,MA,acc = MIA._mem_inf_benchmarks()
            print('filter ratio = {}'.format(1-(data_length+origin_length)/(2*origin_length)))
            print('TP:{TP}, FP:{FP}, FN:{FN},TN={TN}'.format(TP=TP, FP=FP, FN=FN, TN=TN))
            print('R:{R}, P:{P}, F1:{F1},FPR={FPR},MA={MA},acc={acc}'.format(R=R, P=P, F1=F1, FPR=FPR, MA=MA,acc =acc))
            if(index==len(ratio_list)):
                print('finish')
                break


#delete data by score such as risk_score
def score_filter(score_list,shadow_train_performance, shadow_test_performance,
                                       target_train_performance, target_test_performance,risk_score,num_classes):
    for score_thre in score_list:

        #分成对应两部分
        target_train_performance_new_value =[]
        target_train_performance_new_label =[]
        print(' risk_score:{}'.format(score_thre))
        target_train_performance0=target_train_performance[0].tolist()
        target_train_performance1 = target_train_performance[1].tolist()
        origin_length = len(target_train_performance0)
        print('origin_length:{}'.format(origin_length))
        #target_train_performance = (target_train_performance0,target_train_performance1)
        for i in range(len(target_train_performance0)):
            if risk_score[i] >= score_thre:
                target_train_performance_new_value.append(target_train_performance[0][i])
                target_train_performance_new_label.append(target_train_performance[1][i])

        #print(target_train_performance_new_value)
        #print(target_train_performance_new_label)
        #target_train_performance = [np.array(target_train_performance_new_value),np.array(target_train_performance_new_label)]
        print('剔除比例：{}'.format(1-(len(target_train_performance_new_value)+origin_length)/(2*origin_length)))#为什么除以2倍的origin_length 因为比例是算在training+testing中的整体比例

def wrong_classified_nonmem_filter(target_test_performance):
    #choose misclassified non-member
    te_prediction_npy,te_label_npy = target_test_performance
    print(type(te_prediction_npy))
    t_te_corr = (np.argmax(te_prediction_npy, axis=1) == te_label_npy).astype(int)
    return t_te_corr

def distribution_filter(dataset_name='purchase100',dataset=0,sheet_name = '0'):
    #filter data by filter excel files
    fpath = "./ratio_filter/"+dataset_name+"/ratio_filter.xlsx"

    df = pd.read_excel(fpath,sheet_name = sheet_name)
    #index_list = df['index']
    index_list = df.iloc[:,0]
    #print(type(index_list))
    index_list = index_list.values
    index_list = np.array(list(filter(lambda x: not np.isnan(x), [x for x in index_list])))#process nan
    index_list = index_list.astype(int)#
    index_list = index_list.tolist()

    #print(index_list)
    #print(type(index_list))
    mmd_list, xlist, ylist,x_label_list,y_label_list = dataset.mmd_batch()
    #print(f'{len(xlist)=}')
    xlist = xlist[index_list]
    ylist = ylist[index_list]
    x_label_list = x_label_list[index_list]
    y_label_list = y_label_list[index_list]
    #xlist = np.array([x.tolist() for x in xlist])

    xlist = xlist.reshape(len(index_list)*10,dataset.num_classes)
    x_label_list = x_label_list.flatten()
    #print(f'{x_label_list.shape=}')
    ylist = ylist.reshape(len(index_list)*10,dataset.num_classes)
    y_label_list = y_label_list.flatten()
    #print(f'{xlist.size=}')
    #print(f'{xlist.shape=}')
    #print(f'{xlist=}')
    # dataset.shadow_train_performance = (xlist.cpu().detach().numpy(), x_label_list.cpu().detach().numpy())
    # dataset.shadow_test_performance = (ylist.cpu().detach().numpy(), y_label_list.cpu().detach().numpy())

    dataset.target_train_performance = (xlist.cpu().detach().numpy(), x_label_list.cpu().detach().numpy())
    dataset.target_test_performance = (ylist.cpu().detach().numpy(), y_label_list.cpu().detach().numpy())
    # dataset.target_test_performance = (dataset.target_test_performance[0][index_list], dataset.target_test_performance[0][index_list])

    #print('new_distribution_length')

    #print(len(dataset.target_train_performance[0]))