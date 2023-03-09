import os
import numpy as np
import math
import sys
import urllib
import pickle
from torch.autograd import Variable
import argparse
import torch

import filter
from filter import ratio_filter_analysis,score_filter
from MMD import mmd_rbf

import privacy_risk_score_utils
from membership_inference_attacks import black_box_benchmarks
from result_process import find_median,data_analysis

class Dataset(object):

    def __init__(self, path, num_classes = 30):#index用于外界需要打乱时调用
        self.path = path
        self.num_classes = num_classes
        self.tr_prediction_npy = np.load(self.path + 'tr_target.npy', allow_pickle=True)
        self.tr_label_npy = np.load(self.path + 'tr_target_label.npy', allow_pickle=True)
        print(f'{len(self.tr_label_npy)=}')
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
        length = len(self.tr_prediction_npy)
        #print(f'{length=}')
        self.num_classes = len(self.tr_prediction_npy[0])
        print(f'{len(self.target_train_performance[0])=}')
        #print(f'{self.num_classes=}')

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

    def evalue_batch(self,target_train_performance, target_test_performance,
                                   length):
        MIA = black_box_benchmarks(self.shadow_train_performance, self.shadow_test_performance,
                                   target_train_performance, target_test_performance,
                                   num_classes=self.num_classes,
                                   length=length)
        output = MIA._mem_inf_benchmarks()
        return output

    def evalue_have_risk_score(self):
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

    def evalue(self,save_path):
        #print('Perform membership inference attacks!!!')
        length = len(self.tr_prediction_npy)
        #print(f'{self.target_train_performance[0]}')
        MIA = black_box_benchmarks(self.shadow_train_performance, self.shadow_test_performance,
                                   self.target_train_performance, self.target_test_performance,
                                   num_classes=self.num_classes,
                                   length=length)
        TP, FP, FN, TN, R, P, F1, FPR, MA, acc = MIA._mem_inf_benchmarks()
        #print('filter ratio = {}'.format(0))

        with open(save_path, 'a') as f:
            print('R:{R}, P:{P}, F1:{F1},FPR={FPR},MA={MA},acc={acc}'.format(R=R, P=P, F1=F1, FPR=FPR, MA=MA, acc=acc),
                  file=f)

        print('TP:{TP}, FP:{FP}, FN:{FN},TN={TN}'.format(TP=TP, FP=FP, FN=FN, TN=TN))
        print('R:{R}, P:{P}, F1:{F1},FPR={FPR},MA={MA},acc={acc}'.format(R=R, P=P, F1=F1, FPR=FPR, MA=MA, acc=acc))

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
            #use index shuffle
            import random
            random.seed(seed)
            index = [i for i in range(len(X))]
            random.shuffle(index)
            X = X[index]
        else:
            index = None
        X = torch.Tensor(X)
        Y = torch.Tensor(Y)
        return mmd_rbf(X, Y)

    def mmd_batch(self, shuffle=None, seed=0,batch_size = 10,select_index = None,fix_sigma=None):
        '''
        input: dataset
        output: split batch,mmd list

        '''


        #length = len(self.tr_prediction_npy)

        X = self.tr_prediction_npy
        Y = self.te_prediction_npy
        X_label = self.tr_label_npy
        Y_label = self.te_label_npy
        #Z = torch.Tensor(X)
        #print(f'{Z.size()=}')


        # if(select_index):
        #     import random
        #     random.seed(seed)
        #     X = np.array(X)
        #     Y = np.array(Y)
        #     X = X[select_index]
        #     Y = Y[select_index]
        #     #print(f'{X=}')
        #     index = [i for i in range(len(X))]
        #     random.shuffle(index)
        #     X = X[index]
            #print(f'{X=}')

        X = torch.Tensor(X)
        Y = torch.Tensor(Y)
        X_label = torch.Tensor(X_label)
        Y_label = torch.Tensor(Y_label)
        length =len(X)
        #print(f'{X.size()=}')
        xlist = X.split(batch_size, 0)
        ylist = Y.split(batch_size, 0)
        x_label_list = X_label.split(batch_size, 0)
        y_label_list = Y_label.split(batch_size, 0)
        #print(f'{type(xlist)=}')  #tuple 100,10,30 内层仍然是tensor，我们希望将其变为tensor，目前想到的可行方法是，先把内层的tensor脱掉，再最后统一变为tensor.
        # print(f'{(xlist[985].shape)}')

        '''
        xlist = torch.tensor(xlist)
        ylist= torch.tensor(ylist)
        x_label_list = torch.tensor(x_label_list)
        y_label_list = torch.tensor(y_label_list)
        #tuple type can not be tensor by function torch.tensor(), but can use torch.stack
           xlist = torch.tensor(xlist)
            ValueError: only one element tensors can be converted to Python scalars
        
        '''
        #print(f'{type(xlist)=}') #numpy.array
        if (select_index):
            xlist = xlist[select_index]
            ylist = ylist[select_index]
            if shuffle:
                import random
                random.seed(seed)
                # xlist = np.array(xlist)
                # ylist = np.array(ylist)
                index = [i for i in range(len(xlist))]
                random.shuffle(index)
                #print(type(xlist))
                xlist = xlist[index]

        #rint(f'{type(xlist)=}')
        mmd_list = []
        if(select_index):
            for i in range(len(select_index)):
                X, Y = Variable(xlist[i]), Variable(ylist[i])
                mmd_list.append(mmd_rbf(X, Y,fix_sigma=fix_sigma).tolist())
        else:
            for i in range(int(length / batch_size)):
                X, Y = Variable(xlist[i]), Variable(ylist[i])
                mmd_list.append(mmd_rbf(X, Y,fix_sigma=fix_sigma).tolist())
            if not (length == batch_size * len(mmd_list)):
                #print(length)
                #print(batch_size * len(mmd_list))
                #print('warning')
                xlist = xlist[:-1]
                ylist = ylist[:-1]
                x_label_list = x_label_list[:-1]
                y_label_list = y_label_list[:-1]

        xlist = torch.stack(xlist)
        # print(f'{(xlist.shape)=}')
        ylist = torch.stack(ylist)
        x_label_list = torch.stack(x_label_list)
        y_label_list = torch.stack(y_label_list)
        #print(f'{len(xlist)=}')
        return mmd_list, xlist, ylist,x_label_list,y_label_list

    def mmd_batch_movement(self, shuffle=None, seed=0,batch_size = 10,select_index = None,fix_sigma=None):
        '''
        input: dataset
        output: split batch,mmd list
        '''


        #length = len(self.tr_prediction_npy)

        X = self.tr_prediction_npy
        Y = self.te_prediction_npy
        #Z = torch.Tensor(X)
        #print(f'{Z.size()=}')


        # if(select_index):
        #     import random
        #     random.seed(seed)
        #     X = np.array(X)
        #     Y = np.array(Y)
        #     X = X[select_index]
        #     Y = Y[select_index]
        #     #print(f'{X=}')
        #     index = [i for i in range(len(X))]
        #     random.shuffle(index)
        #     X = X[index]
            #print(f'{X=}')

        X = torch.Tensor(X)
        Y = torch.Tensor(Y)
        length =len(X)
        print(f'{X.size()=}')
        xlist = X.split(batch_size, 0)
        ylist = Y.split(batch_size, 0)
        xlist = np.array(xlist)
        ylist= np.array(ylist)
        print(f'{type(xlist)=}')
        if (select_index):
            xlist = xlist[select_index]
            ylist = ylist[select_index]
            if shuffle:
                import random
                random.seed(seed)
                # xlist = np.array(xlist)
                # ylist = np.array(ylist)
                index = [i for i in range(len(xlist))]
                random.shuffle(index)
                print(type(xlist))
                xlist = xlist[index]

        #rint(f'{type(xlist)=}')
        mmd_list = []
        if(select_index):
            for i in range(len(select_index)):
                X, Y = Variable(xlist[i]), Variable(ylist[i])
                mmd_list.append(mmd_rbf(X, Y,fix_sigma=fix_sigma).tolist())
        else:
            for i in range(int(length / batch_size)):
                X1, Y1 = Variable(xlist[i]+xlist[0]), Variable(ylist[i]+ylist[0])
                X2, Y2 = Variable(xlist[0]), Variable(xlist[i]+ylist[i]+ylist[0])
                mmd_list.append(abs(mmd_rbf(X1,X2,fix_sigma=fix_sigma)-mmd_rbf(X1, Y1,fix_sigma=fix_sigma)).tolist())
            if not (length == batch_size * len(mmd_list)):
                print(length)
                print(batch_size * len(mmd_list))
                print('warning')
        return mmd_list, xlist, ylist

    def wrong_classified_nonmem(self):
        '''choose misclassified members and research their mmd '''
        wrong_index = filter.wrong_classified_nonmem_filter(self.target_test_performance)
        #print(f'{wrong_index=}')
        flag = 0
        wrong_data_list = []
        wrong_label_list = []
        for i in range(len(wrong_index)):
            if wrong_index[i] == 0:
                wrong_data_list.append(self.target_test_performance[0][i])
                wrong_label_list.append(self.target_test_performance[1][i])
                flag += 1
            if flag >= 100:
                break

        flag = 0
        correct_data_list = []
        correct_label_list = []
        for i in range(len(wrong_index)):
            if wrong_index[i] == 1:
                correct_data_list.append(self.target_test_performance[0][i])
                correct_label_list.append(self.target_test_performance[1][i])
                flag += 1
            if flag >= 100:
                break
        # choose 100 members
        length = len(wrong_data_list)
        train_data_list = []
        train_label_list = []
        for i in range(length):
                train_data_list.append(self.target_train_performance[0][i])
                train_label_list.append(self.target_train_performance[1][i])
        def evalue(X,Y,train_label_list,test_label_list,save_path= ''):
            X = torch.Tensor(X)
            Y = torch.Tensor(Y)
            X, Y = Variable(X), Variable(Y)
            mmd = mmd_rbf(X,Y)
            print(f'{mmd=}')
            # evalue
            MIA = black_box_benchmarks(self.shadow_train_performance, self.shadow_test_performance,
                                       (np.array(Y),np.array(train_label_list)),(np.array(X), np.array(test_label_list)), self.num_classes, length=length)
            TP, FP, FN, TN,R, P, F1,FPR,MA,acc = MIA._mem_inf_benchmarks()


            #


            print('TP:{TP}, FP:{FP}, FN:{FN},TN={TN}'.format(TP=TP, FP=FP, FN=FN, TN=TN))
            print('R:{R}, P:{P}, F1:{F1},FPR={FPR},MA={MA},acc={acc}'.format(R=R, P=P, F1=F1, FPR=FPR, MA=MA,acc=acc))
        print('wrong:')
        evalue(wrong_data_list,train_data_list,train_label_list,wrong_label_list)
        print('correct:')
        evalue(correct_data_list,train_data_list,train_label_list,correct_label_list)