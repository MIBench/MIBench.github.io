import pandas as pd
filepath = './data_analysis/batch.xlsx'


def output(x1):
    print('mean mmd:' + str(x1['mmd'] / int(len(x) * 0.4)))
    TP = x1['TP']
    FP = x1['FP']
    FN = x1['FN']
    TN = x1['TN']

    print('TP:{TP}, FP:{FP}, FN:{FN},TN={TN}'.format(TP=TP, FP=FP, FN=FN, TN=TN))

    R = TP / (TP + FN)
    P = TP / (TP + FP)
    F1 = 2 * P * R / (P + R)
    FPR = FP / (FP + TN)
    MA = R - FP / (FP + TN)
    acc = acc = (TP + TN) / (TP + TN + FP + FN)
    print('R:{R}, P:{P}, F1:{F1},FPR={FPR},MA={MA},acc={acc}'.format(R=R, P=P, F1=F1, FPR=FPR, MA=MA, acc=acc))

#
datasetname = ['purchase100','location30','texas100']
# datasetname = ['purchase100']
for sheet_name in datasetname:
    print(sheet_name)
    df = pd.read_excel(filepath,sheet_name=sheet_name)
    df = df.sort_values(by="mmd")
    # print(f'{df.head=}')
    list = ['TP','FP','FN','TN']
    x = df
    # top %0~40%, %60~%100
    x_small = x[:int(len(x)*0.4)]
    x_big = x[int(len(x)*0.6):]
    # print(x_big.head())
    # print(x_small.head())
    x1 = x_small.sum()
    x2 = x_big.sum()
    # print(f'{x1=}')
    # print(f'{x2=}')

    output(x1)
    output(x2)