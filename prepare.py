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
from utils import *


def softmax_by_row(logits, T=1.0):
    mx = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp((logits - mx) / T)
    denominator = np.sum(exp, axis=-1, keepdims=True)
    return exp / denominator


def prepare_model_performance(shadow_model, shadow_train_loader, shadow_test_loader,
                              target_model, target_train_loader, target_test_loader):
    def _model_predictions(model, dataloader):
        return_outputs, return_labels = [], []

        for (inputs, labels) in dataloader:
            return_labels.append(labels.numpy())
            outputs = model.forward(inputs.cuda())
            return_outputs.append(softmax_by_row(outputs.data.cpu().numpy()))
        return_outputs = np.concatenate(return_outputs)
        return_labels = np.concatenate(return_labels)
        return (return_outputs, return_labels)

    shadow_train_performance = _model_predictions(shadow_model, shadow_train_loader)
    shadow_test_performance = _model_predictions(shadow_model, shadow_test_loader)

    target_train_performance = _model_predictions(target_model, target_train_loader)
    target_test_performance = _model_predictions(target_model, target_test_loader)
    return shadow_train_performance, shadow_test_performance, target_train_performance, target_test_performance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run membership inference attacks')
    parser.add_argument('--dataset', type=str, default='texas', help='purchase or texas')
    # parser.add_argument('--model-dir', type=str, default='./pretrained_models/purchase_defended',
    #                     help='directory of target model')
    parser.add_argument('--model-dir', type=str, default='./pretrained_models/texas',
                        help='directory of target model')
    parser.add_argument('--save_path', type=str, default='/home/zhaoqy/pyproject/MIA_risk_score/saved_predictions/texas100/',
                        help='directory of target model')


    parser.add_argument('--batch-size', type=int, default=100, help='batch size of data loader')
    args = parser.parse_args()

    saved_path =args.save_path

    if args.dataset == 'purchase':
        model = PurchaseClassifier(num_classes=100)
        model = torch.nn.DataParallel(model).cuda()
        shadow_train_loader, shadow_test_loader, \
        target_train_loader, target_test_loader = prepare_purchase_data(batch_size=args.batch_size)
    else:
        model = TexasClassifier(num_classes=100)
        model = torch.nn.DataParallel(model).cuda()
        shadow_train_loader, shadow_test_loader, \
        target_train_loader, target_test_loader = prepare_texas_data(batch_size=args.batch_size)

    checkpoint = torch.load(args.model_dir)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    shadow_train_performance, shadow_test_performance, target_train_performance, target_test_performance = \
        prepare_model_performance(model, shadow_train_loader, shadow_test_loader,
                                  model, target_train_loader, target_test_loader)

    # target
    tr_prediction_npy, tr_label_npy = target_train_performance
    te_prediction_npy, te_label_npy = target_test_performance

    np.save(saved_path + 'tr_target', np.array(tr_prediction_npy))
    np.save(saved_path + 'tr_target_label', np.array(tr_label_npy))
    np.save(saved_path + 'te_target', np.array(te_prediction_npy))
    np.save(saved_path + 'te_target_label', np.array(te_label_npy))

    # shadow
    tr_prediction_npy, tr_label_npy = shadow_train_performance
    te_prediction_npy, te_label_npy = shadow_test_performance

    np.save(saved_path + 'tr_shadow', np.array(tr_prediction_npy))
    np.save(saved_path + 'tr_shadow_label', np.array(tr_label_npy))
    np.save(saved_path + 'te_shadow', np.array(te_prediction_npy))
    np.save(saved_path + 'te_shadow_label', np.array(te_label_npy))

    print('process successful')