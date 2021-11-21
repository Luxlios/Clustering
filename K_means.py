# -*- coding = utf-8 -*-
# @Time : 2021/11/21 13:26
# @Author : Luxlios
# @File : K_means.py
# @Software : PyCharm

import pandas as pd
import numpy as np
import cv2
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

# Kmeans聚类算法(硬聚类)
def K_means(data, K):
    # 输入数据data和簇数K
    num = data.shape[0]
    _class = np.zeros([num], np.int)  # 保存每一个元素的分类

    # 初始化中心点，随机选取k个索引，得到k个初始中心点
    rand = np.random.random(size=K)  # np.random.randint(size)
    rand = rand * num
    rand = np.floor(rand).astype(int)
    center = data[rand]
    # print(rand)

    # 主体循环，通过初始中心对于所有数据点分类
    # 对于这些分类，每一类计算每一个维度的均值，得到新的中心点
    # 循环下去直到中心点不发生变化
    change = True  # 中心点是否变化的flag
    while change:
        distance = np.zeros([num, K])
        for i in range(num):
            for j in range(K):
                # L2距离,得到样本与所有中心的距离
                distance[i, j] = np.sqrt(np.sum((data[i, :] - center[j, :]) ** 2))

        for i in range(num):
            _class[i] = np.argmin(distance[i, :])  # 得到最近距离的索引

        change = False
        for i in range(K):
            cluster = data[_class == i]  # 得到分类索引为i的所有数据
            center_new = np.mean(cluster, axis=0)
            if np.sum(np.abs(center[i] - center_new), axis=0) > 10:
                center[i] = center_new
                change = True
    # 返回聚类中心和样本类别
    return center, _class


# 展示聚类结果的函数
# iris数据集有四维，只选用前三维数据进行展示
def show(data, center, _class, k, name):
    # 输入数据，聚类中心，分类结果，簇数
    plt.figure()
    color = ['r', 'g', 'b', 'c', 'y', 'm']
    num = data.shape[0]
    picture = plt.subplot(111, projection='3d')
    for i in range(k):
        picture.scatter(center[i][0], center[i][1], center[i][2], c=color[i], marker='x')
    for i in range(num):
        picture.scatter(data[i][0], data[i][1], data[i][2], c=color[_class[i]], marker='.')
    plt.title(name)

if __name__ == '__main__':
    # Iris数据集聚类
    data = pd.read_csv('iris.data', header=None)  # 有header会把第一行数据当列名
    data = np.array(data)
    x = data[:, [0, 1, 2, 3]]  # 数据
    y = data[:, 4]  # 标签
    x = generalization(x)  # 标准化
    center1, class1 = K_means(x, 3)
    show(x, center1, class1, 3, 'K_means')

    # 图像聚类分割
    img = cv2.imread('D:\hh.png')
    _img = img.copy()
    _img = _img[:, :, ::-1]
    plt.figure()
    plt.subplot(111), plt.imshow(_img), plt.title('hh'), plt.axis('off')

    # 将图像像素矩阵读成一列
    pixel = img.reshape(-1, 3)
    pixel = np.float32(pixel)
    # K_means聚类
    center3, class3 = K_means(pixel, 4)
    center4, class4 = K_means(pixel, 6)
    center5, class5 = K_means(pixel, 8)

    # 聚类结果展示
    k_picture3 = class3.reshape(img.shape[0], img.shape[1])
    k_picture4 = class4.reshape(img.shape[0], img.shape[1])
    k_picture5 = class5.reshape(img.shape[0], img.shape[1])
    plt.figure()
    plt.subplot(131), plt.imshow(k_picture3), plt.title('k=4'), plt.axis('off')
    plt.subplot(132), plt.imshow(k_picture4), plt.title('k=6'), plt.axis('off')
    plt.subplot(133), plt.imshow(k_picture5), plt.title('k=8'), plt.axis('off')
    plt.show()