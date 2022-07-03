import torch
import sys
import random
from torch.utils.data import DataLoader
import numpy as np


# 划分数据集
class download_data(object):
    def __init__(self, batch_size=4):
        # self.root = root  # root = "C:\\Users\\wxlou\\Desktop\\program\\code\\lfchu\\data\\train.csv"
        self.root = "C:\\Users\\wxlou\\Desktop\\program\\code\\lfchu\\data\\train.csv"
        self.batch_size = batch_size
        self.train_data = None
        self.test_data = None
        self.load_dataset()  # 初始化数据集

    def load_dataset(self):
        image_data = []                         # 存放打包的数据，为以后划分训练集合测试集做准备

        raw_datasets = np.loadtxt(self.root, dtype=np.str, delimiter=",")[1:,1:]     # 加载数据
        datasets = get_datasets(raw_datasets[:,:-1])                # 仅仅处理原始数据，不处理标签
        labels = [int(one_data[-1]) for one_data in raw_datasets]        # 加载标签
        # print(labels)

        for raw_data,label in zip(datasets,labels):
            data = resize(raw_data)
            data2 = torch.tensor(data,dtype=torch.float32)
            image_data.append((data2,label))
        # print(image_data[10])
        # sys.exit()
        self.split_dataset(image_data)

    def split_dataset(self, image_data):
        # 按照4:1划分数据集为训练集和测试集
        dataset = list(range(0, len(image_data)))
        random.shuffle(dataset)                                 # 随机化
        sampling_train = dataset[:int(len(dataset) * 0.8)]
        sampling_test = dataset[int(len(dataset) * 0.8):]

        train = torch.utils.data.Subset(image_data, sampling_train)
        self.train_data = DataLoader(dataset=train, batch_size=self.batch_size, shuffle=True)     # 封装训练集
        test = torch.utils.data.Subset(image_data, sampling_test)
        self.test_data = DataLoader(dataset=test, batch_size=self.batch_size, shuffle=True)             # 封装测试集
        print("训练集长度：", len(sampling_train))
        print("测试集长度：", len(sampling_test))

    def get_test_dataset(self):
        return self.test_data

    def get_train_dataset(self):
        return self.train_data


def get_datasets(dataset,way=None):
    data = dataset.astype(float)
    # 如果way是None，表示不处理
    # 如果way是"L2"，表示L2正则变换
    # 如果way是"Z-score"，表示处理成均值为0，方差为1的数据
    if way == None:
        return data

def resize(data,size=(60,100)):
    # 将一个数据变成三维数据：[1,x,y]的形式
    # size表示修改后的尺寸: 默认为1x60x100
    new_data = []
    mid_data = []
    for i in range(size[0]):                 # 将数据一行一行提取出来
        mid_data.append(data[size[1]*i:size[1]*(1+i)])
    new_data.append(mid_data)
    return new_data



