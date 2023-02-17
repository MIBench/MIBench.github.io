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

def data_analysis(input_list,path,name,if_save=False):
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
    if(if_save):
        plt.savefig(path + name)
    plt.show()

    return 0


import numpy as np
import matplotlib.pyplot as plt


class make_distribute():
    '''


    example:
    import numpy as np
    import matplotlib.pyplot as plt

    realdata = (np.random.normal(loc=500, scale=80, size=1000))
    a = make_distribute()
    y, x = a.run(realdata, 5)

    plt.plot(x, y)
    plt.show()
    '''

    def __init__(self):
        return None

    def run(self, datalist, interval):
        # arg: datalist is a numpy.array, interval is int
        datalist.sort()
        index = 0
        flag = 0
        distribute_list = []
        # print(index)
        # i 遍历间隔空间
        for i in range(int(min(datalist)), int(max(datalist)), interval):

            # print('i:{}'.format(i))

            if (len(datalist) == index + 1):
                break

            while (True):
                # 小于第i个区间上限个数,要遍历整个空间
                # print('index:{}'.format(datalist[index]))
                if (datalist[index] <= (i + interval)):

                    if (len(datalist) == index + 1):
                        break
                    index += 1
                else:
                    # print('index:{}'.format(datalist[index]))
                    # print('flag{}'.format(flag))
                    # print('index-flag:{}'.format(index-flag))
                    # print('toobig{}'.format(index))
                    distribute_list.append((index - flag))
                    flag = index
                    # print('flag{}'.format(flag))
                    break

        x = list(range(len(distribute_list)))
        for i in range(len(x)):
            x[i] = x[i] * 5 + min(datalist)
        return distribute_list, x

# def gene_distibution():
