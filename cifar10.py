import os
import numpy as np
import math
import sys
import urllib
import pickle

from matplotlib import pyplot as plt
from torch.autograd import Variable
import argparse
import torch
from filter import ratio_filter,score_filter
from MMD import mmd_rbf

import privacy_risk_score_utils
from membership_inference_attacks import black_box_benchmarks

class cifar10(object):

    def __init__(self, path, num_classes):
        self.path = path
        self.num_classes = num_classes
        self.tr_prediction_npy = np.load(self.path + 'tr_target.npy', allow_pickle=True)
        self.tr_label_npy = np.load(self.path + 'tr_target_label.npy', allow_pickle=True)

        self.te_prediction_npy = np.load(self.path + 'te_target.npy', allow_pickle=True)
        self.te_label_npy = np.load(self.path + 'te_target_label.npy', allow_pickle=True)

        # shadow_model
        self.sh_tr_prediction_npy = np.load(self.path + 'tr_shadow.npy', allow_pickle=True)
        self.sh_tr_label_npy = np.load(self.path + 'tr_shadow_label.npy', allow_pickle=True)

        self.sh_te_prediction_npy = np.load(self.path + 'te_shadow.npy', allow_pickle=True)
        self.sh_te_label_npy = np.load(self.path + 'te_shadow_label.npy', allow_pickle=True)

        self.shadow_train_performance = (self.sh_tr_prediction_npy, self.sh_tr_label_npy)
        self.shadow_test_performance = (self.sh_te_prediction_npy, self.sh_te_label_npy)
        self.target_train_performance = (self.tr_prediction_npy, self.tr_label_npy)
        self.target_test_performance = (self.te_prediction_npy, self.te_label_npy)

    def gene_risk_score(self):
        # target_model
        print('Perform membership inference attacks!!!')
        length = len(self.tr_prediction_npy)
        MIA = black_box_benchmarks(self.shadow_train_performance, self.shadow_test_performance,
                                   self.target_train_performance, self.target_test_performance,
                                   num_classes=self.num_classes,
                                   length=length)
        MIA._mem_inf_benchmarks()

        risk_score = privacy_risk_score_utils.calculate_risk_score(MIA.s_tr_m_entr, MIA.s_te_m_entr, MIA.s_tr_labels,
                                                                   MIA.s_te_labels,
                                                                   MIA.t_tr_m_entr, MIA.t_tr_labels)
        # print('risk_score_shape: {a}, score: {b}'.format(a = risk_score.shape,b = risk_score))
        # np.save('cifar10_regularization_risk_score.npy',risk_score)
        np.save(self.path + '/risk_score.npy', risk_score)
        self.risk_score_path = self.path + '/risk_score.npy'

    # def show_risk_score(self):
    #     risk_score = np.load(self.risk_score_path)
    #     print('risk score: {}'.format(risk_score))

    def evalue(self):
        risk_score = np.load(self.risk_score_path)
        print('length of risk score: {}'.format(len(risk_score)))
        # print(risk_score)
        ratio_list = [0.1, 0.2, 0.3, 0.4]
        # target_model

        print('Perform membership inference attacks!!!')
        # print(len(self.target_train_performance[0]))
        # print(self.target_train_performance[0])
        ratio_filter(ratio_list, self.shadow_train_performance, self.shadow_test_performance,
                     self.target_train_performance, self.target_test_performance, risk_score, self.num_classes)

    def self_score_filter(self):
        risk_score = np.load(self.risk_score_path)
        # print('length of risk score: {}'.format(len(risk_score)))
        score_list = []
        for i in range(30, 70):
            score_list.append(i / 100)
            # target_model

        print('Perform membership inference attacks!!!')
        score_filter(score_list, self.shadow_train_performance, self.shadow_test_performance,
                     self.target_train_performance, self.target_test_performance, risk_score, self.num_classes)

    def mmd(self, shuffle=None, seed=0):
        # self.tr_prediction_npy = np.load(self.path + 'tr_target.npy', allow_pickle=True)
        # self.tr_label_npy = np.load(self.path + 'tr_target_label.npy', allow_pickle=True)
        #
        # self.te_prediction_npy = np.load(self.path + 'te_target.npy', allow_pickle=True)
        # self.te_label_npy = np.load(self.path + 'te_target_label.npy', allow_pickle=True)

        X = self.tr_prediction_npy
        Y = self.te_prediction_npy
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(X)
            np.random.shuffle(Y)
        X = torch.Tensor(X)
        Y = torch.Tensor(Y)
        return mmd_rbf(X, Y)

    def mmd_batch(self, shuffle=None, seed=0,batch_size = 10):
        '''
        输入数据集返回分batch返回mmd列表
        '''


        length = len(self.tr_prediction_npy)

        X = self.tr_prediction_npy
        Y = self.te_prediction_npy
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(X)
            np.random.shuffle(Y)

        X = torch.Tensor(X)
        Y = torch.Tensor(Y)

        xlist = X.split(batch_size, 0)
        ylist = Y.split(batch_size, 0)

        mmd_list = []
        for i in range(int(length / batch_size)):
            X, Y = Variable(xlist[i]), Variable(ylist[i])
            mmd_list.append(mmd_rbf(X, Y).tolist())
        if not (length == batch_size * len(mmd_list)):
            print(length)
            print(batch_size * len(mmd_list))
            print('warning')
        return mmd_list, xlist, ylist

CIFAR = cifar10(num_classes = 10,path = './saved_predictions/cifar10/')
for i in range(50):
    batch_size = 10
    mmd_list,_,_ = CIFAR.mmd_batch(shuffle=1,seed = i,batch_size = batch_size)

    #print(mmd_list)
    #画出mmd_list的分布图
    #

    # num_bins = 20
    # plt.hist(mmd_list, num_bins)
    # path ='./mmd_distribution/batch_size={}/'.format(batch_size)
    #
    # if not(os.path.exists(path)):
    #     os.makedirs(path)
    # plt.savefig(path+'pic-{}.png'.format(i))
    # plt.cla()

# mmd1 = 0
# for i in range(9):
#     mmd1 = mmd1 + mmd_list[i]/10
# print(mmd1)
# mmd2 = 0
# for i in range(10):
#     mmd2 = mmd2 + mmd_list[i+9] / 10
# print(mmd2)
# for i in range(81):
#     if abs(mmd1+mmd_list[i+19] /10 - mmd2) <= 0.02:
#         print(mmd1+mmd_list[i+19] /10)
#         print(i)
#
# xcatlist1 = []
# for i in range(9):
#     xcatlist1.append(xlist[i])
# xcatlist1.append(xlist[5+19])
# xoutputs1 = torch.cat(xcatlist1,0)
# #print(xoutputs1)
#
# ycatlist1 = []
# for i in range(9):
#     ycatlist1.append(ylist[i])
# ycatlist1.append(ylist[5+19])
# youtputs1 = torch.cat(ycatlist1,0)
# #print(youtputs1)
#
# xcatlist2 = []
# for i in range(10):
#     xcatlist2.append(xlist[i+9])
# xoutputs2 = torch.cat(xcatlist2,0)
# #print(xoutputs2)
#
# ycatlist2 = []
# for i in range(10):
#     ycatlist2.append(ylist[i+9])
# youtputs2 = torch.cat(ycatlist2,0)
# #print(youtputs2)
#
#
# print(mmd_rbf(xoutputs1,youtputs1))
# print(mmd_rbf(xoutputs2,youtputs2))