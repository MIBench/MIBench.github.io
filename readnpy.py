# -*- coding: gbk -*-
import numpy as np
import argparse

parser = argparse.ArgumentParser()
#��ͬ�����ݼ���û��ǵ������ļ��У���Ϊ�ļ��Ƚ϶࣬�������ڵ��úͲ鿴

#parser.add_argument('-p',"--path", help="path", type=str,default='./saved_predictions/cifar10/tr_target.npy')
#parser.add_argument('-p',"--path", help="path", type=str,default=r'C:\Users\20784\PycharmProjects\MIA\MIA_risk_score\data\texas\random_r_texas100.npy')
parser.add_argument('-p',"--path", help="path", type=str,default=r'C:\Users\20784\PycharmProjects\MIA\MIA_risk_score\data\texas\shuffle_index.npz')


args = parser.parse_args()

dataset = np.load(args.path,allow_pickle=True)
print(dataset.files)
print(len(dataset.files))
print(dataset['x'].shape)
print(dataset['x'][0])
# print(dataset)
# print(dataset.shape)