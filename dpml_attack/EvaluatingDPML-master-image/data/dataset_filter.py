from __future__ import print_function
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import torch
from tensorflow import keras
from torch import nn



import argparse
import os




import torch.nn.functional as F


import torch
import torch.nn.parallel

from texas_undefended import Texas
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
import numpy as np
import tarfile
import urllib
from purchase_undefended import PurchaseClassifier

torch.cuda.is_available()

np.set_printoptions(threshold=np.inf)

import sys
import os
import argparse
import pickle
import numpy as np



sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from core.attack import proposed_membership_inference
from core.attack import evaluate_proposed_membership_inference
from core.utilities import log_loss

np.set_printoptions(threshold=np.inf)

'''提取原始的数据'''




def dataset_filter(dataset_name='location30',sheet_name = '0'):


    fpath = "./ratio_filter/" + dataset_name + "/ratio_filter.xlsx"

    df = pd.read_excel(fpath, sheet_name=sheet_name)
    index_list = df.iloc[:,0]
    #index_list = df['index']
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

    '''文本数据的'''
    # #目标模型训练集挑选后的结果
    # X_target = X[new_index_list]
    # Y_target = Y[new_index_list]
    #
    # #影子模型训练集挑选后的结果
    # X_shadow = X[shadow_index_list]
    # Y_shadow = Y[shadow_index_list]
    #
    # #测试集挑选后的结果
    # X_test = X[new_test_index_list]
    # Y_test = Y[new_test_index_list]
    # print()


    '''图像数据'''
    
    X_target = x_train[new_index_list]
    Y_target = y_train[new_index_list]# [[0.0.1.0][1.0.0.0]]
    # print(Y_target)
    # Y_target = np.argmax(Y_target,axis=1){4,5,4,8,1,}
    #影子模型训练集挑选后的结果
    X_shadow = x_train[new_index_list]
    Y_shadow = y_train[new_index_list]
    # Y_shadow = np.argmax(Y_shadow, axis=1)
    # print(Y_shadow)
    #测试集挑选后的结果
    X_test = x_test[new_index_list]
    Y_test = y_test[new_index_list]
    # Y_test = np.argmax(Y_test, axis=1)

    '''添加相关训练和攻击过程'''
    true_x = np.concatenate((X_target, X_test), axis=0)
    true_y = np.concatenate((Y_target, Y_test), axis=0)# (5000*8)
    # print("ture_x:", true_x)
    #print("ture_y:", true_y)
    v_ture_x = np.concatenate((X_shadow, X_test), axis=0)
    v_ture_y = np.concatenate((Y_shadow, Y_test), axis=0)

    '''添加相关训练和攻击过程'''

    # 攻击需要的参数

    # 开始攻击
    if dataset_name == 'purchase100':

        print('-' * 10 + 'TRAIN TARGET' + '-' * 10 + '\n')

        # 获取目标模型参数
        classifier = PurchaseClassifier()
        checkpoint = torch.load('./models/purchase_undefended/model_best.pth')
        classifier.load_state_dict(checkpoint['state_dict'], False)
        classifier = torch.nn.DataParallel(classifier).cuda()
        # 训练目标模型
        # 处理数据集
        set_size = len(true_x)
        # 计算pred_y
        pred_y = np.zeros(set_size)
        per_instance_loss = np.zeros(set_size)
        for i in range(set_size):
            x = (true_x[i]).astype(np.float32)
            x = torch.tensor(x)
            # print(x)
            x = x.cuda()
            output = classifier(x)[0]
            print("output: ", output)
            y_pred = torch.argmax(output)
            print("y_pred:", y_pred)
            pred_y[i] = y_pred
            y = (true_y[i]).astype(np.float32)
            y = torch.tensor(y)
            y = y.cuda()
            per_instance_loss[i] = F.cross_entropy(output, y.long())
        print("pred_y:", pred_y)

        in_target = np.ones(len(X_target))
        out_target = np.zeros(len(X_test))
        membership = np.concatenate((in_target, out_target), axis=0)
        print("membership:", membership)
        in_shadow = np.ones(len(X_shadow))
        out_target = np.zeros(len(X_test))
        v_membership = np.concatenate((in_shadow, out_target), axis=0)
        test_classes = true_y

        # 计算训练集加上测试集每个数据的损失，pred_y是预测的y
        print("per_instance_loss:", per_instance_loss)
        # # Yeom's membership inference attack when only train_loss is known
        # yeom_mi_outputs_1 = yeom_membership_inference(per_instance_loss, membership, train_loss)
        # # Yeom's membership inference attack when both train_loss and test_loss are known - Adversary 2 of Yeom et al.
        # yeom_mi_outputs_2 = yeom_membership_inference(per_instance_loss, membership, train_loss, test_loss)
        #
        # # Shokri's membership inference attack
        # shokri_mi_outputs = shokri_membership_inference(args, pred_y, membership, test_classes)

        # Proposed membership inference attacks
        proposed_mi_outputs = proposed_membership_inference(v_ture_x, v_ture_y, v_membership, true_x, true_y,
                                                            classifier, per_instance_loss,
                                                            args)
        fpr, fnr, precision, recall, accuracy, f1_score = evaluate_proposed_membership_inference(per_instance_loss,
                                                                                                 membership,
                                                                                                 proposed_mi_outputs,
                                                                                                 fpr_threshold=0.01)
        # evaluate_proposed_membership_inference(per_instance_loss, membership, proposed_mi_outputs, fpr_threshold=0.01,
        #                                        per_class_thresh=True)

        # 将度量值输出
        print(fpr, fnr, precision, recall, accuracy, f1_score)
        # 写入文件
        f = open('./purchase_result/' + dataset_name + '_' + sheet_name + 'result.txt', 'w')
        f.write("fpr: " + str(fpr) + " fnr: " + str(fnr) + " precision: " + str(precision) + " recall: " + str(
            recall) + " accuracy: " + str(accuracy) + " f1_score: " + str(f1_score) + "\n")
        f.close()

        # fpr, fnr, precision, recall, accuracy, f1_score = evaluate1(train_losses,out_losses,0.0001,attack_base='loss')
        # # 将度量值输出
        # print(fpr, fnr, precision, recall, accuracy, f1_score)
        #
        # # 写入文件
        # f = open('./purchase_loss/'+dataset_name+'_'+sheet_name+'_loss.txt','w')
        # f.write("train_losses: "+str(train_losses)+"\n")
        # f.write("out_losses:"+str(out_losses)+"\n")
        # f.close()
        #
        # f = open('./purchase_result/'+dataset_name + '_' + sheet_name + 'result.txt', 'w')
        # f.write("fpr: "+str(fpr)+" fnr: "+str(fnr)+" precision: "+str(precision)+" recall: "+str(recall)+" accuracy: "+str(accuracy)+" f1_score: "+str(f1_score)+"\n")
        # f.close()
    elif dataset_name == 'texas100':
        print('-' * 10 + 'TRAIN TARGET' + '-' * 10 + '\n')

        # 获取目标模型参数
        classifier = Texas()
        checkpoint = torch.load('./models/texas_undefended/model_best.pth')
        classifier.load_state_dict(checkpoint['state_dict'], False)
        classifier = torch.nn.DataParallel(classifier).cuda()
        # 训练目标模型
        # 处理数据集
        set_size = len(true_x)
        # 计算pred_y
        pred_y = np.zeros(set_size)
        per_instance_loss = np.zeros(set_size)
        for i in range(set_size):
            x = (true_x[i]).astype(np.float32)
            x = torch.tensor(x)
            # print(x)
            x = x.cuda()
            output = classifier(x)[0]
            print("output: ", output)
            y_pred = torch.argmax(output)

            print("y_pred:", y_pred)
            pred_y[i] = y_pred
            y = (true_y[i]).astype(np.float32)
            y = torch.tensor(y)
            y = y.cuda()
            per_instance_loss[i] = F.cross_entropy(output, y.long())
        print("pred_y:", pred_y)

        in_target = np.ones(len(X_target))
        out_target = np.zeros(len(X_test))
        membership = np.concatenate((in_target, out_target), axis=0)
        print("membership:", membership)
        in_shadow = np.ones(len(X_shadow))
        out_target = np.zeros(len(X_test))
        v_membership = np.concatenate((in_shadow, out_target), axis=0)
        test_classes = true_y

        # 计算训练集加上测试集每个数据的损失，pred_y是预测的y
        print("per_instance_loss:", per_instance_loss)
        # # Yeom's membership inference attack when only train_loss is known
        # yeom_mi_outputs_1 = yeom_membership_inference(per_instance_loss, membership, train_loss)
        # # Yeom's membership inference attack when both train_loss and test_loss are known - Adversary 2 of Yeom et al.
        # yeom_mi_outputs_2 = yeom_membership_inference(per_instance_loss, membership, train_loss, test_loss)
        #
        # # Shokri's membership inference attack
        # shokri_mi_outputs = shokri_membership_inference(args, pred_y, membership, test_classes)

        # Proposed membership inference attacks
        proposed_mi_outputs = proposed_membership_inference(v_ture_x, v_ture_y, v_membership, true_x, true_y,
                                                            classifier, per_instance_loss,
                                                            args)
        fpr, fnr, precision, recall, accuracy, f1_score = evaluate_proposed_membership_inference(per_instance_loss,
                                                                                                 membership,
                                                                                                 proposed_mi_outputs,
                                                                                                 fpr_threshold=0.01)
        # evaluate_proposed_membership_inference(per_instance_loss, membership, proposed_mi_outputs, fpr_threshold=0.01,
        #                                        per_class_thresh=True)

        # 将度量值输出
        print(fpr, fnr, precision, recall, accuracy, f1_score)
        # 写入文件
        f = open('./texas_result/' + dataset_name + '_' + sheet_name + 'result.txt', 'w')
        f.write("fpr: " + str(fpr) + " fnr: " + str(fnr) + " precision: " + str(precision) + " recall: " + str(
            recall) + " accuracy: " + str(accuracy) + " f1_score: " + str(f1_score) + "\n")
        f.close()
    else:
         classifier= keras.models.load_model(WEIGHTS_PATH)
         # 处理数据集
         set_size = len(true_x)
         # 计算pred_y
         true_y = torch.tensor(true_y)
         x = torch.tensor(true_x)
         pred_y = classifier(x)#[[1,2,3][5,4,1]]
         #print(pred_y)# (5000*8)
         pred_y_=tf.nn.softmax(pred_y)#[[1,0,0],[0,0,1]]
         per_instance_loss = tf.nn.softmax_cross_entropy_with_logits(logits=true_y,labels=pred_y_)
         #print("pred_y:",pred_y)
         print("per_instance_loss:",per_instance_loss)#(5000,)


         in_target = np.ones(len(X_target))
         out_target = np.zeros(len(X_test))
         membership = np.concatenate((in_target, out_target), axis=0)
         print("membership:", membership)
         in_shadow = np.ones(len(X_shadow))
         out_target = np.zeros(len(X_test))
         v_membership = np.concatenate((in_shadow, out_target), axis=0)


         # 计算训练集加上测试集每个数据的损失，pred_y是预测的y
         # # Yeom's membership inference attack when only train_loss is known
         # yeom_mi_outputs_1 = yeom_membership_inference(per_instance_loss, membership, train_loss)
         # # Yeom's membership inference attack when both train_loss and test_loss are known - Adversary 2 of Yeom et al.
         # yeom_mi_outputs_2 = yeom_membership_inference(per_instance_loss, membership, train_loss, test_loss)
         #
         # # Shokri's membership inference attack
         # shokri_mi_outputs = shokri_membership_inference(args, pred_y, membership, test_classes)

         # Proposed membership inference attacks
         proposed_mi_outputs = proposed_membership_inference(v_ture_x, v_ture_y, v_membership, true_x, true_y,
                                                             classifier, per_instance_loss,
                                                             args)
         fpr, fnr, precision, recall, accuracy, f1_score,ma = evaluate_proposed_membership_inference(per_instance_loss,
                                                                                                  membership,
                                                                                                  proposed_mi_outputs,
                                                                                                  fpr_threshold=0.2)
         # evaluate_proposed_membership_inference(per_instance_loss, membership, proposed_mi_outputs, fpr_threshold=0.01,
         #                                        per_class_thresh=True)

         # 将度量值输出
         print(fpr, fnr, precision, recall, accuracy, f1_score)
         # 写入文件
         f = open('./result/' + dataset_name + '_' + sheet_name + 'result.txt', 'w')
         f.write("fpr: " + str(fpr) + " fnr: " + str(fnr) + " precision: " + str(precision) + " recall: " + str(
             recall) + " accuracy: " + str(accuracy) + " f1_score: " + str(f1_score) + "ma: "+str(ma)+"\n")
         f.close()
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default='cifar_100')
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--use_cpu', type=int, default=0)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--save_data', type=int, default=0)
    # target and shadow model configuration
    parser.add_argument('--n_shadow', type=int, default=5)
    parser.add_argument('--target_data_size', type=int, default=int(1e4))
    parser.add_argument('--target_test_train_ratio', type=float, default=2.0)
    parser.add_argument('--target_model', type=str, default='nn')
    parser.add_argument('--target_learning_rate', type=float, default=0.01)
    parser.add_argument('--target_batch_size', type=int, default=200)
    parser.add_argument('--target_n_hidden', type=int, default=256)
    parser.add_argument('--target_epochs', type=int, default=100)
    parser.add_argument('--target_l2_ratio', type=float, default=1e-8)
    parser.add_argument('--target_clipping_threshold', type=float, default=1.0)
    parser.add_argument('--target_privacy', type=str, default='no_privacy')
    parser.add_argument('--target_dp', type=str, default='dp')
    parser.add_argument('--target_epsilon', type=float, default=0.5)
    parser.add_argument('--target_delta', type=float, default=1e-5)
    # attack model configuration
    parser.add_argument('--attack_model', type=str, default='nn')
    parser.add_argument('--attack_learning_rate', type=float, default=0.01)
    parser.add_argument('--attack_batch_size', type=int, default=100)
    parser.add_argument('--attack_n_hidden', type=int, default=64)
    parser.add_argument('--attack_epochs', type=int, default=100)
    parser.add_argument('--attack_l2_ratio', type=float, default=1e-6)
    # Merlin's noise parameters
    parser.add_argument('--attack_noise_type', type=str, default='gaussian')
    parser.add_argument('--attack_noise_coverage', type=str, default='full')
    parser.add_argument('--attack_noise_magnitude', type=float, default=0.01)

    # specify datafile names
    parser.add_argument('--data_id', type=int, default=-1)

    # parse configuration
    args = parser.parse_args()
    print(vars(args))
# dataset_names = ['purchase100','texas100','CH_MNIST','CIFAR100','CIFAR10','imagenet']
#dataset_names = ['texas100']
MODEL = "ResNet50"
dataset_names = ['imagenet','CIFAR100','CIFAR10','CH_MNIST']
for dataset_name in dataset_names:
    WEIGHTS_PATH = "../tuxiang/Weights/Target_Model/{}_{}.hdf5".format(dataset_name, MODEL)
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


    else:
        data_path = './image_datasets/' + dataset_name + '/Target_data/' + dataset_name
        # (x_train, y_train), (x_test, y_test) =
        x_train = np.load(data_path + '_x_train.npy')
        # print(f'{x_train[0].shape=}')
        y_train = np.load(data_path + '_y_train.npy')
        x_test = np.load(data_path + '_x_test.npy')
        y_test = np.load(data_path + '_y_test.npy')

    print(dataset_name)
    fpath = "./ratio_filter/" + dataset_name + "/ratio_filter.xlsx"
    reader = pd.ExcelFile(fpath)
    sheet_names = reader.sheet_names
    for sheet_name in sheet_names:
        print(sheet_name)
        dataset_filter(dataset_name=dataset_name,sheet_name = sheet_name)