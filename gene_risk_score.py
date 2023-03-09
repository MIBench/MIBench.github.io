import os
import numpy as np
import math
import sys
import urllib
import pickle
import argparse
#sys.path.append('../')

import privacy_risk_score_utils
from membership_inference_attacks import black_box_benchmarks



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run membership inference attacks')
    parser.add_argument('--dataset', type=str, default='cifar10', help='location or texas')
    #parser.add_argument('--predictions-dir', type=str, default='./saved_predictions',help='directory of saved predictions')
    #parser.add_argument('--defended', type=int, default=0, help='1 means defended; 0 means natural')
    #parser.add_argument('--length', type=int, default=1000, help='')
    parser.add_argument('--path', type=str, default='./saved_predictions/cifar10/', help='data path')

    args = parser.parse_args()
    if(args.dataset == 'cifar10'):
        num_classes = 10
    # elif(args.dataset == 'purchase'):
    #     num_class =
    # elif (args.dataset == 'texas'):
    #     num_class =


    #target_model
    tr_prediction_npy = np.load(args.path+ 'tr_target.npy',allow_pickle=True)
    tr_label_npy= np.load(args.path + 'tr_target_label.npy',allow_pickle=True)

    te_prediction_npy = np.load(args.path + 'te_target.npy', allow_pickle=True)
    te_label_npy = np.load(args.path + 'te_target_label.npy', allow_pickle=True)

    #shadow_model
    sh_tr_prediction_npy = np.load(args.path + 'tr_shadow.npy', allow_pickle=True)
    sh_tr_label_npy = np.load(args.path + 'tr_shadow_label.npy', allow_pickle=True)

    sh_te_prediction_npy = np.load(args.path + 'te_shadow.npy', allow_pickle=True)
    sh_te_label_npy = np.load(args.path + 'te_shadow_label.npy', allow_pickle=True)

    shadow_train_performance = (sh_tr_prediction_npy,sh_tr_label_npy)
    shadow_test_performance = (sh_te_prediction_npy,sh_te_label_npy)
    target_train_performance =(tr_prediction_npy,tr_label_npy)
    target_test_performance =(te_prediction_npy,te_label_npy)

    print('Perform membership inference attacks!!!')

    MIA = black_box_benchmarks(shadow_train_performance, shadow_test_performance,
                               target_train_performance, target_test_performance, num_classes=num_classes)
    MIA._mem_inf_benchmarks()


    risk_score = privacy_risk_score_utils.calculate_risk_score(MIA.s_tr_m_entr, MIA.s_te_m_entr, MIA.s_tr_labels, MIA.s_te_labels,
                                      MIA.t_tr_m_entr, MIA.t_tr_labels)
    #print('risk_score_shape: {a}, score: {b}'.format(a = risk_score.shape,b = risk_score))
    # np.save('cifar10_regularization_risk_score.npy',risk_score)
    np.save(args.path+'/risk_score.npy', risk_score)