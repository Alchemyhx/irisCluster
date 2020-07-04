#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 21:31:49 2020
@author: Hexin Yuan 19210240055
"""
from sklearn.cluster import SpectralClustering
import numpy as np
import math


def import_data_from_iris(filename):
    data = []
    cluster_raw = []

    with open(str(filename), 'r') as f:
        for line in f:
            line_temp = line.strip().split()
            line_temp_dumy = []
            for j in range(0, len(line_temp) - 1):
                line_temp_dumy.append(float(line_temp[j]))
            data.append(line_temp_dumy)
            cluster_raw.append(line_temp[j + 1])

    return data, cluster_raw


def calculate_accuracy(labels, k_num):
    right = 0
    for k in range(0, k_num):
        checker = [0, 0, 0]
        for i in range(0, 50):
            checker[int(labels[i + 50 * k])] += 1
        right += max(checker)
    return right


if __name__ == '__main__':
    data, cluster_raw = import_data_from_iris("iris.dat")
    print("iris.dat数据：\n", data)
    dataArr = np.array(data)

    sc = SpectralClustering(n_clusters=3)
    sc.fit(dataArr)

    print("谱聚类结果：")
    for i in range(3):
        print(i, "类包含的样本")
        for j in range(len(data)):
            if int(sc.labels_[j]) == i:
                print(j, data[j])

    right_num = calculate_accuracy(sc.labels_, 3)
    print("错误率：", 1 - right_num / len(data))

