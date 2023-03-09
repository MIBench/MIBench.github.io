#构造所谓的分布
'''
输入应该是排好序的batch，然后随便搞个均匀分布（数据量初步定为总数据集的50%）可以先看看
也要看一下整个mmd的分布
'''
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import font_manager
import random




def find_median(mmd_list):
    '''
    先排序，再一遍找到位置.输入为numpy.array
    '''
    mmd_list = np.sort(mmd_list)
    median = mmd_list[int(len(mmd_list)/2)]
    return median

def data_analysis(input_list):
    length  = (max(input_list) - min(input_list))

    # 计算组数
    d = 0.05  # 组距
    num_bins = int((length) // d)

    # 设置图形大小
    plt.figure(figsize=(20, 8), dpi=80)
    plt.hist(input_list, num_bins)

    print(f'{min(input_list)}')
    # 设置x轴刻度
    plt.xticks(np.arange(min(input_list), max(input_list) + d, d))

    # 设置网格
    plt.grid(alpha=0.4)
    plt.show()
    return 0

# def gene_distibution():
