'''
1.下面分别是CIFAR10\CIFAR100\CH_MNIST\imagenet 的数据构造过程，
  大家可以直接将数据处理部分函数直接提出来进行使用：
  我会将数据处理部分和原数据分别输出，大家都可以核对测试。

'''
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os
import torchvision
from tqdm import tqdm
import torch as torch
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds

def load_imagenet(model_mode):   #针对imagnet数据集的操作。
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('不是目标模型或者影子模型！')
    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    '''
    将tiny-imagenet-200压缩包放在目录下，解压使用即可
    '''
    dataset = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train', transform=transform)
    generator=torch.Generator().manual_seed(0)
    train_data,test_data = torch.utils.data.random_split(dataset, [90000, 10000],generator=generator)
    train_loader = DataLoader(train_data, batch_size=1000, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False, num_workers=0)
    x_train = []
    y_train = []
    for x_, y_ in train_loader:
        for x in x_:
            x = x.transpose(0, 2)
            x_train.append(x.numpy())
        for y in y_:
            y_train.append(int(y.numpy()))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test, y_test = [], []
    for x_, y_ in test_loader:
        for x in x_:
            x = x.transpose(0, 2)
            x_test.append(x.numpy())
        for y in y_:
            y_test.append(int(y.numpy()))
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=200)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=200)
    if model_mode == "TargetModel":  #在这里针对不同模型进行数据的划分选取数据量为20000
        (x_train, y_train), (x_test, y_test) = (x_train[80000:], y_train[80000:]), \
                                               (x_test, y_test)
    elif model_mode == "ShadowModel":
        (x_train, y_train), (x_test, y_test) = (x_train[:10000], y_train[:10000]), \
                                               (x_train[10000:20000], y_train[10000:20000])
    m_train = np.ones(y_train.shape[0])
    m_test = np.zeros(y_test.shape[0])
    member = np.r_[m_train, m_test]
    return (x_train,y_train),(x_test,y_test),member

def load_CH_MNIST(model_mode):   #CH_MNIST数据集的处理部分
    '''
    这里需要下载“colorectal_histology”数据集，运行需要等待
    '''
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('不是目标模型或者影子模型！')
    #初始化数据
    tf.random.set_seed(1)
    images, labels = tfds.load('colorectal_histology', split='train', batch_size=-1, as_supervised=True)

    x_train, x_test, y_train, y_test = train_test_split(images.numpy(), labels.numpy(), train_size=0.5,
                                                        random_state=1 if model_mode == 'TargetModel' else 3,
                                                        stratify=labels.numpy())

    x_train = tf.image.resize(x_train, (32, 32))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=8)
    m_train = np.ones(y_train.shape[0])
    x_test = tf.image.resize(x_test, (32, 32))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=8)
    m_test = np.zeros(y_test.shape[0])
    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member

def load_CIFAR(model_mode):     #CIFAR100
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('不是目标模型或者影子模型！')
    tf.random.set_seed(1)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    if model_mode == "TargetModel":
        (x_train, y_train), (x_test, y_test) = (x_train[40000:50000], y_train[40000:50000]), \
                                               (x_test, y_test)
    elif model_mode == "ShadowModel":
        (x_train, y_train), (x_test, y_test) = (x_train[:10000], y_train[:10000]), \
                                               (x_train[10000:20000], y_train[10000:20000])

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
    m_train = np.ones(y_train.shape[0])
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
    m_test = np.zeros(y_test.shape[0])
    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member

def load_CIFAR10(model_mode):
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('不是目标模型或者影子模型！')
    tf.random.set_seed(1)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if model_mode == "TargetModel":
        (x_train, y_train), (x_test, y_test) = (x_train[40000:50000], y_train[40000:50000]), \
                                               (x_test, y_test)
    elif model_mode == "ShadowModel":
        (x_train, y_train), (x_test, y_test) = (x_train[:10000], y_train[:10000]), \
                                               (x_train[10000:20000], y_train[10000:20000])
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    m_train = np.ones(y_train.shape[0])
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    m_test = np.zeros(y_test.shape[0])
    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member
