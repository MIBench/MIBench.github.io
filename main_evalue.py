import numpy as np
import argparse
from dataset import Dataset
import pandas as pd
import numpy as np
import result_process
import filter
#import privacy_risk_score_utils
from membership_inference_attacks import black_box_benchmarks

#原始数据


def evalue():
    dataset = Dataset(num_classes=num_classes, path=path)
    dataset.gene_risk_score()
    dataset.evalue_have_risk_score()


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


def wrong_classified_nonmem_mmd():
    #choose misclassified non-member and research their mmd
    dataset = Dataset(num_classes=num_classes, path=path)
    dataset.wrong_classified_nonmem()


def distribution_filter_(dataset_name = ' location30'):
    '''
    filter by excel files
    attack

    '''
    #dataset = Dataset(num_classes=num_classes, path=path)
    #fpath = "./distribution_filter/" + dataset_name + "/distribution.xlsx"
    fpath = "./ratio_filter/" + dataset_name + "/ratio_filter.xlsx"
    reader = pd.ExcelFile(fpath)
    sheet_names = reader.sheet_names
    for sheet_name in sheet_names:
        dataset = Dataset(path=path)
        #print(sheet)
        filter.distribution_filter(args.dataset,dataset,sheet_name)
        save_path = './risk_score_result/' + dataset_name
        with open(save_path, 'a') as f:
            print('sheet_name= '+sheet_name,file =f )
        dataset.evalue(save_path)
        del dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run membership inference attacks')
    #parser.add_argument('--dataset', type=str, default='cifar10', help='location30 or cifar10')
    parser.add_argument('--dataset', type=str, default='purchase100', help='location30 or cifar10')
    #parser.add_argument('--predictions-dir', type=str, default='./saved_predictions',help='directory of saved predictions')
    #parser.add_argument('--defended', type=int, default=0, help='1 means defended; 0 means natural')
    #parser.add_argument('--length', type=int, default=1000, help='')
    #parser.add_argument('--path', type=str, default='./saved_predictions/cifar10/', help='data path')
    #parser.add_argument('--risk_score_path', type=str, default='./saved_predictions/cifar10/risk_score.npy', help='risk score path')

    # args = parser.parse_args()
    # datasetname = ['purchase100','location30','texas100']
    # writer = pd.ExcelWriter('./data_analysis/batch.xlsx')
    # for i in datasetname:
    #     args.dataset = i
    #     print('attack dataset:{}'.format(args.dataset))
    #     if(args.dataset == 'cifar10'):
    #         num_classes = 10
    #         path = './saved_predictions/cifar10/'
    #     elif (args.dataset == 'location30'):
    #         num_classes = 30
    #         path = './saved_predictions/location30/'
    #     elif (args.dataset == 'texas100'):
    #         num_classes = 100
    #         path = './saved_predictions/texas100/'
    #     elif (args.dataset == 'purchase100'):
    #         num_classes = 100
    #         path = './saved_predictions/purchase100/'
    #
    #     batch_analysis()
    #
    # writer.save()

    #evalue()


    args = parser.parse_args()
    datasetname = ['purchase100','location30','texas100','CH_MNIST','CIFAR100','CIFAR10','imagenet']
    #datasetname = ['CH_MNIST', 'CIFAR100', 'CIFAR10', 'imagenet']
    #datasetname = ['location30']
    #writer = pd.ExcelWriter('./data_analysis/batch.xlsx')
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
        else:
            path = './saved_predictions/'+ args.dataset+'/'


        distribution_filter_(args.dataset)







