from get_data import download_data
from model import MNIST_RNN,FMNIST_RNN
from torch.autograd import Variable
from torch import nn
import torch
import sys
import copy
import torch.optim as optim
import argparse
import time
import os



def main():
    print('初始化数据集...')
    data_loader = download_data()

    test_data = data_loader.get_test_dataset()          # 加载测试集
    train_data = data_loader.get_train_dataset()        # 加载训练集
    # print(test_data)

    print("初始化模型...")
    model = FMNIST_RNN()            # 产生训练模型
    optimizer = optim.SGD(params=model.parameters(), lr=0.002, momentum=0.9)      # 定义SGD优化器

    acc = []
    losses = []
    for epoch in range(200):
        print("\n第{}轮：".format(epoch))

        # 本地训练
        model.train()           # 表示模型可以训练了
        for data,target in train_data:
            # print(data.size())
            # print(target)
            # sys.exit()
            data, target = Variable(data), Variable(target)             # 变成可以计算梯度的形式，可以忽略

            optimizer.zero_grad()               # 梯度清零
            output = model(data)                # 输入数据得到输出

            loss = nn.CrossEntropyLoss()(output, target)    # 计算损失
            loss.backward()
            optimizer.step()                    # 更新


        model.eval()            # 表示模型不可训练
        with torch.no_grad():   # torch.no_grad()是一个上下文管理器，用来禁止梯度的计算
            test_correct = 0
            test_loss = 0
            for data, target in test_data:
                data, target = Variable(data), Variable(target)

                output = model(data)
                l = nn.CrossEntropyLoss()(output, target).item()
                test_loss += l
                test_correct += (torch.sum(torch.argmax(output, dim=1) == target)).item()
            acc.append(test_correct/len(test_data.dataset))
            losses.append(test_loss/len(test_data.dataset))

        #return test_loss, test_correct / len(self.test_data.dataset)

        print("精度：", acc[epoch-1])
        print("损失：", losses[epoch-1])

    print(acc)




if __name__ == "__main__":
    main()
