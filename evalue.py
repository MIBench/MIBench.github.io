import numpy as np
import argparse
from dataset import Dataset
import pandas as pd
import numpy as np
import result_process
#import privacy_risk_score_utils
#from membership_inference_attacks import black_box_benchmarks

#原始数据
#path = './saved_predictions/cifar10_natural_predictions/'

#正则化
# path = './saved_predictions/cifar10_regularization_predictions/'

def evalue():
    dataset = Dataset(num_classes=num_classes, path=path)
    dataset.gene_risk_score()
    dataset.evalue()


def batch_analysis():
    dataset = Dataset(num_classes=num_classes, path=path)
    # print(dataset.mmd())
    mmd_list, xlist, ylist, x_label_list, y_label_list = dataset.mmd_batch()
    movement_mmd_list, _, _ = dataset.mmd_batch_movement()
    indexlist = [i for i in range(len(xlist))]
    metric_list = [[], [], [], [], [], [], [], [], [], []]
    print(f'{len(indexlist)=}')
    print(f'{len(mmd_list)=}')
    print(f'{len(movement_mmd_list)=}')
    dataframe = pd.DataFrame(
        {'index': indexlist, 'mmd': mmd_list, 'movement_mmd': movement_mmd_list})  # 'A'是columns，对应的是list

    for j in range(len(xlist)):
        # shadow_train_performance = (xlist[j].cpu().detach().numpy(), x_label_list[j].cpu().detach().numpy())
        # shadow_test_performance = (ylist[j].cpu().detach().numpy(), y_label_list[j].cpu().detach().numpy())
        target_train_performance = (xlist[j].cpu().detach().numpy(), x_label_list[j].cpu().detach().numpy())
        target_test_performance = (ylist[j].cpu().detach().numpy(), y_label_list[j].cpu().detach().numpy())
        length = len(ylist[j])
        # print(type(target_test_performance[0]))
        output = dataset.evalue_batch(target_train_performance, target_test_performance,
                                      length=length)
        for k in range(len(output)):
            metric_list[k].append(output[k])

    namelist = ['TP', 'FP', 'FN', 'TN', 'R', 'P', 'F1', 'FPR', 'MA', 'acc']
    for name in range(len(namelist)):
        dataframe[namelist[name]] = metric_list[name]
    dataframe.to_excel(writer, sheet_name=args.dataset)
    #dataframe.to_excel(excel_writer='./data_analysis/batch.xlsx',sheet_name=args.dataset)
    # all_batch_num = len(mmd_list)
    # result_process.data_analysis(mmd_list, name=args.dataset+'movement_mmd', path='./data_analysis/',if_save=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run membership inference attacks')
    #parser.add_argument('--dataset', type=str, default='cifar10', help='location30 or cifar10')
    parser.add_argument('--dataset', type=str, default='purchase100', help='location30 or cifar10')
    #parser.add_argument('--predictions-dir', type=str, default='./saved_predictions',help='directory of saved predictions')
    #parser.add_argument('--defended', type=int, default=0, help='1 means defended; 0 means natural')
    #parser.add_argument('--length', type=int, default=1000, help='')
    #parser.add_argument('--path', type=str, default='./saved_predictions/cifar10/', help='data path')
    #parser.add_argument('--risk_score_path', type=str, default='./saved_predictions/cifar10/risk_score.npy', help='risk score path')

    args = parser.parse_args()
    datasetname = ['purchase100','location30','texas100']
    writer = pd.ExcelWriter('./data_analysis/batch.xlsx')
    for i in datasetname:
        args.dataset = i
        print('attack dataset:{}'.format(args.dataset))
        if(args.dataset == 'cifar10'):
            num_classes = 10
            path = './saved_predictions/cifar10/'
        elif (args.dataset == 'location30'):
            num_classes = 30
            path = './saved_predictions/location30/'
        elif (args.dataset == 'texas100'):
            num_classes = 100
            path = './saved_predictions/texas100/'
        elif (args.dataset == 'purchase100'):
            num_classes = 100
            path = './saved_predictions/purchase100/'

        batch_analysis()

    writer.save()

    #evalue()










#以下均为测试代码功能，以及其他用途的测试函数

    # def mytest_U():#观察所谓的batch的均匀分布是否存在
    #     #dataset = Dataset(path,num_classes)
    #     dataset = Dataset(num_classes=num_classes, path=path)
    #     #print(dataset.mmd())
    #     mmd_list, _, _ = dataset.mmd_batch()
    #     all_batch_num = len(mmd_list)
    #     result_process.data_analysis(mmd_list, name=args.dataset, path='./data_analysis/')
    #
    #     mmd_list_index = np.argsort(mmd_list)#对索引值排序
    #     #print(mmd_list_index)
    #     mmd_list= np.array(mmd_list)
    #     mmd_list = mmd_list[mmd_list_index]
    #     #print(f'{mmd_list=}')
    #     index = []
    #     #mylist = []#每个bins取10个batch
    #     bins = 0.5
    #     for i in range(all_batch_num):#0.5~0.8步长0.1
    #         if mmd_list[i]>=bins:
    #             for j in range(5):
    #                 index.append(mmd_list_index[i+j])
    #                 #mylist.append(mmd_list_index[i+j])
    #             bins += 0.1
    #         if bins>0.8:
    #             break
    #     #index = np.array(index)
    #     print(f'{index=}')
    #     mmd_list_new_unshuffle, _, _ = dataset.mmd_batch(select_index= index)
    #     mmd_list_new_shuffle, _, _ = dataset.mmd_batch(shuffle=True, select_index=index)
    #     result_process.data_analysis(mmd_list_new_unshuffle, name=args.dataset+'unshuffle', path='./data_analysis/result_process_')
    #     result_process.data_analysis(mmd_list_new_shuffle, name=args.dataset+'shuffle', path='./data_analysis/result_process_')
    #
    # mytest_U()




