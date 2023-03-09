import os
import numpy as np
import math
import sys
import urllib
import pickle
import torch
from filter import ratio_filter
import input_data_class
import privacy_risk_score_utils
from membership_inference_attacks import black_box_benchmarks
from MMD import mmd_rbf
from torch.autograd import Variable

class location30(object):

    def __init__(self,path,num_classes):
        self.path = path
        self.num_classes = num_classes

    def gene_risk_score(self):
        dataset = 'location'
        input_data = input_data_class.InputData(dataset=dataset)
        (x_target, y_target, l_target) = input_data.input_data_attacker_evaluate()
        npz_data = np.load(self.path + 'target_predictions.npz')

        target_predictions = npz_data['tc_output']
        target_train_performance = (target_predictions[l_target == 1], y_target[l_target == 1].astype('int32'))
        target_test_performance = (target_predictions[l_target == 0], y_target[l_target == 0].astype('int32'))

        (x_shadow, y_shadow, l_shadow) = input_data.input_data_attacker_adv1()
        npz_data = np.load(self.path + 'shadow_predictions.npz')

        shadow_predictions = npz_data['tc_output']

        shadow_train_performance = (shadow_predictions[l_shadow == 1], y_shadow[l_shadow == 1].astype('int32'))
        shadow_test_performance = (shadow_predictions[l_shadow == 0], y_shadow[l_shadow == 0].astype('int32'))

        print('Perform membership inference attacks!!!')
        length = len(target_train_performance[0])
        MIA = black_box_benchmarks(shadow_train_performance, shadow_test_performance,
                                   target_train_performance, target_test_performance, num_classes=self.num_classes,length=length)
        MIA._mem_inf_benchmarks()

        risk_score = privacy_risk_score_utils.calculate_risk_score(MIA.s_tr_m_entr, MIA.s_te_m_entr, MIA.s_tr_labels,
                                                                   MIA.s_te_labels,
                                                                   MIA.t_tr_m_entr, MIA.t_tr_labels)
        print('risk_score_shape: {a}, score: {b}'.format(a=risk_score.shape, b=risk_score))

        np.save(self.path + '/risk_score.npy', risk_score)
        self.risk_score_path = self.path + '/risk_score.npy'


    def evalue(self):
        #risk_score = np.load(self.risk_score_path)
        #print('length of risk score: {}'.format(len(risk_score)))
        # print(risk_score)
        ratio = [0.1, 0.2, 0.3, 0.4]
        input_data = input_data_class.InputData(dataset='location')
        (x_target, y_target, l_target) = input_data.input_data_attacker_evaluate()
        npz_data = np.load(self.path + 'target_predictions.npz')

        target_predictions = npz_data['tc_output']
        target_train_performance = (target_predictions[l_target == 1], y_target[l_target == 1].astype('int32'))
        target_test_performance = (target_predictions[l_target == 0], y_target[l_target == 0].astype('int32'))

        (x_shadow, y_shadow, l_shadow) = input_data.input_data_attacker_adv1()
        npz_data = np.load(self.path + 'shadow_predictions.npz')

        shadow_predictions = npz_data['tc_output']

        shadow_train_performance = (shadow_predictions[l_shadow == 1], y_shadow[l_shadow == 1].astype('int32'))
        shadow_test_performance = (shadow_predictions[l_shadow == 0], y_shadow[l_shadow == 0].astype('int32'))

        # target
        tr_prediction_npy, tr_label_npy = target_train_performance
        te_prediction_npy, te_label_npy = target_test_performance

        np.save(self.path + 'tr_target', np.array(tr_prediction_npy))
        np.save(self.path + 'tr_target_label', np.array(tr_label_npy))
        np.save(self.path + 'te_target', np.array(te_prediction_npy))
        np.save(self.path + 'te_target_label', np.array(te_label_npy))

        # shadow
        tr_prediction_npy, tr_label_npy = shadow_train_performance
        te_prediction_npy, te_label_npy = shadow_test_performance

        np.save(self.path + 'tr_shadow', np.array(tr_prediction_npy))
        np.save(self.path + 'tr_shadow_label', np.array(tr_label_npy))
        np.save(self.path + 'te_shadow', np.array(te_prediction_npy))
        np.save(self.path + 'te_shadow_label', np.array(te_label_npy))
        print('save successful!!!')
        # print(len(target_train_performance[0]))
        # print(target_train_performance[0])
        #ratio_filter(ratio, shadow_train_performance, shadow_test_performance,
        #            target_train_performance, target_test_performance, risk_score, self.num_classes)

    def mmd(self):
        input_data = input_data_class.InputData(dataset='location')
        (x_target, y_target, l_target) = input_data.input_data_attacker_evaluate()
        npz_data = np.load(self.path + 'target_predictions.npz')

        target_predictions = npz_data['tc_output']
        target_train_performance = (target_predictions[l_target == 1], y_target[l_target == 1].astype('int32'))
        target_test_performance = (target_predictions[l_target == 0], y_target[l_target == 0].astype('int32'))
        X = torch.Tensor(target_train_performance[0])
        Y = torch.Tensor(target_test_performance[0])
        # X = (target_train_performance[0])
        # Y = (target_test_performance[0])
        X, Y = Variable(X), Variable(Y)
        return mmd_rbf(X,Y)

A = location30(r'C:\Users\20784\PycharmProjects\MIA\MIA_risk_score\saved_predictions\location30\\',num_classes=30)
A.evalue()