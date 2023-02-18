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

# from core.attack import proposed_membership_inference
# from core.attack import evaluate_proposed_membership_inference
# from core.utilities import log_loss

np.set_printoptions(threshold=np.inf)

'''提取原始的数据'''

def \
        evaluate1(train_losses, out_losses, threshold, attack_base=None):
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
    ma = recall - fpr
    return fpr,fnr, precision, recall, accuracy,f1_score,ma


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
    Y_target = y_train[new_index_list]
    X_target = (X_target).astype(np.float32)# [[0.0.1.0][1.0.0.0]]
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
    X_test = (X_test).astype(np.float32)
    # X_test = np.array(X_test)
    # Y_test = np.array(Y_test)
    # Y_test = np.argmax(Y_test, axis=1)

    '''添加相关训练和攻击过程'''

    '''添加相关训练和攻击过程'''

    # 攻击需要的参数

    # 开始攻击
    if dataset_name == 'purchase100':

       return

        # 计算训练集加上测试集每个数据的损失，pred_y是预测的
        # # Yeom's membership inference attack when only train_loss is known
        # yeom_mi_outputs_1 = yeom_membership_inference(per_instance_loss, membership, train_loss)
        # # Yeom's membership inference attack when both train_loss and test_loss are known - Adversary 2 of Yeom et al.
        # yeom_mi_outputs_2 = yeom_membership_inference(per_instance_loss, membership, train_loss, test_loss)
        #
        # # Shokri's membership inference attack
        # shokri_mi_outputs = shokri_membership_inference(args, pred_y, membership, test_classes)

        # Proposed membership inference attacks
        # proposed_mi_outputs = proposed_membership_inference(v_ture_x, v_ture_y, v_membership, true_x, true_y,
        #                                                     classifier, per_instance_loss,
        #                                                     args)
        # fpr, fnr, precision, recall, accuracy, f1_score = evaluate_proposed_membership_inference(per_instance_loss,
        #                                                                                          membership,
        #                                                                                          proposed_mi_outputs,
        #                                                                                          fpr_threshold=0.01)
        # evaluate_proposed_membership_inference(per_instance_loss, membership, proposed_mi_outputs, fpr_threshold=0.01,
        #                                        per_class_thresh=True)

        # # 将度量值输出
        # print(fpr, fnr, precision, recall, accuracy, f1_score)
        # # 写入文件
        # f = open('./purchase_result/' + dataset_name + '_' + sheet_name + 'result.txt', 'w')
        # f.write("fpr: " + str(fpr) + " fnr: " + str(fnr) + " precision: " + str(precision) + " recall: " + str(
        #     recall) + " accuracy: " + str(accuracy) + " f1_score: " + str(f1_score) + "\n")
        # f.close()

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
       return
        # # Yeom's membership inference attack when only train_loss is known
        # yeom_mi_outputs_1 = yeom_membership_inference(per_instance_loss, membership, train_loss)
        # # Yeom's membership inference attack when both train_loss and test_loss are known - Adversary 2 of Yeom et al.
        # yeom_mi_outputs_2 = yeom_membership_inference(per_instance_loss, membership, train_loss, test_loss)
        #
        # # Shokri's membership inference attack
        # shokri_mi_outputs = shokri_membership_inference(args, pred_y, membership, test_classes)

        # Proposed membership inference attacks
        # proposed_mi_outputs = proposed_membership_inference(v_ture_x, v_ture_y, v_membership, true_x, true_y,
        #                                                     classifier, per_instance_loss,
        #                                                     args)
        # fpr, fnr, precision, recall, accuracy, f1_score = evaluate_proposed_membership_inference(per_instance_loss,
        #                                                                                          membership,
        #                                                                                          proposed_mi_outputs,
        #                                                                                          fpr_threshold=0.01)
        # evaluate_proposed_membership_inference(per_instance_loss, membership, proposed_mi_outputs, fpr_threshold=0.01,
        #                                        per_class_thresh=True)

        # # 将度量值输出
        # print(fpr, fnr, precision, recall, accuracy, f1_score)
        # # 写入文件
        # f = open('./texas_result/' + dataset_name + '_' + sheet_name + 'result.txt', 'w')
        # f.write("fpr: " + str(fpr) + " fnr: " + str(fnr) + " precision: " + str(precision) + " recall: " + str(
        #     recall) + " accuracy: " + str(accuracy) + " f1_score: " + str(f1_score) + "\n")
        # f.close()
    else:
         classifier= keras.models.load_model(WEIGHTS_PATH)

         # 处理数据集
         # 计算pred_y
         in_y = torch.tensor(Y_target)
         out_y = torch.tensor(Y_test)
         in_x = torch.tensor(X_target)
         out_x = torch.tensor(X_test)
         in_pred_y = classifier(in_x)#[[1,2,3][5,4,1]]
         out_pred_y = classifier(out_x)  # [[1,2,3][5,4,1]]
         #print(pred_y)# (5000*8)
         in_pred_y_=tf.nn.softmax(in_pred_y)#[[1,0,0],[0,0,1]]
         out_pred_y_ = tf.nn.softmax(out_pred_y)  # [[1,0,0],[0,0,1]]
         in_loss = tf.nn.softmax_cross_entropy_with_logits(logits=in_y,labels=in_pred_y_)
         out_loss = tf.nn.softmax_cross_entropy_with_logits(logits=out_y, labels=out_pred_y_)
         #print("pred_y:",pred_y)
         # print("in_loss:",in_loss)#(5000,)
         # print("out_loss:",out_loss)

         # in_y_a = torch.tensor(Y_target)
         # out_y_a = torch.tensor(Y_test)
         # in_x_a = torch.tensor(X_target)
         # out_x_a = torch.tensor(X_test)

         print("走到这里了")

         # 这里有问题
         # in_x_a = list(in_x)
         # print("走到这里了11111111")
         # out_x_a = list(out_x)
         # print("走到这里了22222222")
         classifier_shadow = keras.models.load_model(WEIGHTS_PATH_SHADOW)
         print("走到这里333333")
         in_pred_y_a = classifier_shadow(in_x)
         print("走到这里了444444")
         out_pred_y_a = classifier_shadow(out_x)  # [[1,2,3][5,4,1]]
         # print(pred_y)# (5000*8)
         in_pred_y_a = tf.nn.softmax(in_pred_y_a)  # [[1,0,0],[0,0,1]]
         out_pred_y_a = tf.nn.softmax(out_pred_y_a)  # [[1,0,0],[0,0,1]]
         in_loss_attack = tf.nn.softmax_cross_entropy_with_logits(logits=in_y, labels=in_pred_y_a)
         out_loss_attack = tf.nn.softmax_cross_entropy_with_logits(logits=out_y, labels=out_pred_y_a)
         # print("pred_y:",pred_y)
         # print("in_loss:", in_loss_attack)  # (5000,)
         #print("out_loss:", out_loss_attack)
         print("走到这里了")

         in_losses = in_loss - in_loss_attack
         out_losses = out_loss - out_loss_attack
         print("in_losses:",in_losses)
         print("out_losses:",out_losses)
         fpr, fnr, precision, recall, accuracy, f1_score,ma = evaluate1(in_losses, out_losses, 0.0001,
                                                                     attack_base='loss')
         # 将度量值输出






         # 计算训练集加上测试集每个数据的损失，pred_y是预测的y
         # # Yeom's membership inference attack when only train_loss is known
         # yeom_mi_outputs_1 = yeom_membership_inference(per_instance_loss, membership, train_loss)
         # # Yeom's membership inference attack when both train_loss and test_loss are known - Adversary 2 of Yeom et al.
         # yeom_mi_outputs_2 = yeom_membership_inference(per_instance_loss, membership, train_loss, test_loss)
         #
         # # Shokri's membership inference attack
         # shokri_mi_outputs = shokri_membership_inference(args, pred_y, membership, test_classes)

         # Proposed membership inference attacks

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
dataset_names = ['imagenet','CH_MNIST','CIFAR100','CIFAR10']
for dataset_name in dataset_names:
    WEIGHTS_PATH = "../tuxiang/Weights/Target_Model/{}_{}.hdf5".format(dataset_name, MODEL)
    WEIGHTS_PATH_SHADOW = "../tuxiang/Weights/Shadow_Model/{}_{}_sha.hdf5".format(dataset_name, MODEL)
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