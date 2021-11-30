# -*- coding = utf-8 -*-
# @Time : 2021/11/21 13:33
# @Author : Luxlios
# @File : FCM.py
# @Software : PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 数据标准化函数
def generalization(data):
    num = data.shape[1]   # 读取列数，对每一列标准化
    _data = np.zeros([num,1], np.float)
    for i in range(num):
        _data = data[:,i]
        _range = np.max(_data)-np.min(_data)
        data[:,i] = (_data-np.min(_data))/_range
    return data


# FCM聚类算法（软聚类）
# 采用初始化隶属度矩阵，计算聚类中心，计算代价行数后再更新隶属度矩阵的策略
def FCM(data, K, alpha):
    # 输入数据data和簇数K和柔性参数alpha
    num = data.shape[0]

    # 每一行均匀分布并且和为1，对隶属度矩阵u初始化
    u = np.random.random([num, K])
    # np.sum后化作行array向量，因此需要添加一个低位轴转化为列array
    u = np.divide(u, np.sum(u, axis=1)[:, np.newaxis])

    change = True  # 隶属度矩阵与阈值大小关系的flag
    while change:
        # 目标函数（总的欧式距离点乘隶属矩阵，所有元素之和）最小，
        # 用lagrange乘数法得到隶属矩阵和聚类中心的更新公式
        center = np.divide(np.dot((u ** alpha).T, data), np.sum(u ** alpha, axis=0)[:, np.newaxis])  # dot为矩阵乘法

        change = False
        # 计算样本与聚类中心的距离
        distance = np.zeros([num, K])
        for i in range(num):
            for j in range(K):
                distance[i, j] = np.linalg.norm(data[i, :] - center[j, :], ord=2)  # L2距离

        # 更新隶属度矩阵
        u_new = np.zeros([num, K])
        for i in range(num):
            for j in range(K):
                u_new[i, j] = 1. / np.sum((distance[i, j] / distance[i, :]) ** (2 / (alpha - 1)))

        # 判断隶属度矩阵变化与阈值的比较
        if np.sum(abs(u_new - u)) > 10:
            change = True
            u = u_new
    # 返回聚类中心和样本隶属度矩阵
    return center, u_new

# 展示聚类结果的函数
# iris数据集有四维，只选用前三维数据进行展示
def show(data, center, _class, k, name):
    # 输入数据，聚类中心，分类结果，簇数
    color = ['r', 'g', 'b', 'c', 'y', 'm']
    num = data.shape[0]
    picture = plt.subplot(111, projection='3d')
    for i in range(k):
        picture.scatter(center[i][0], center[i][1], center[i][2], c=color[i], marker='x')
    for i in range(num):
        picture.scatter(data[i][0], data[i][1], data[i][2], c=color[_class[i]], marker='.')
    plt.title(name)
    plt.show()

if __name__ == '__main__':
    # IRIS数据集读取
    data = pd.read_csv('.\data\iris.data', header=None)  # 有header会把第一行数据当列名
    data = np.array(data)
    x = data[:, [0, 1, 2, 3]]  # 数据
    y = data[:, 4]  # 标签
    x = generalization(x)  # 标准化
    center2, u = FCM(x, 3, 2)
    # 选取隶属度最大的作为种类，方便后续评价效果
    class2 = np.argmax(u, axis=1)
    show(x, center2, class2, 3, 'FCM')
