# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 14:12:24 2017

@author: yli
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:08:38 2017

@author: li_yu
"""

import numpy as np
import os

path = 'train_data.txt'
save_path = 'result_for_give_feature/'
feature_index=[6,7,8,9,10,11,12,13]
num = 10000

if not os.path.exists(save_path):
    os.makedirs(save_path)


def convert_to_float(datapath,num):
    train=[]
    f = open(datapath,'r')
    line = f.readline()
    for i in range(num-1):
        item=[]
        line = f.readline()
        line = line.strip('\n')
        line=line.split()
        for element in feature_index:
            if line[element] == 'NULL':
                item.append(0.)
            else:
                item.append(float(line[element]))
        item.append(float(line[6]))
        train.append(item)
    return train

train_data = convert_to_float(path,num)


def mean_var(dataMatrix):
    mean_list = []
    var_list = []
    for i in range(len(dataMatrix[0])-1):
        datalist = [x[i] for x in dataMatrix]
        mean = sum(datalist)/len(dataMatrix)
 
        tmp_for_var = [(x-mean)**2 for x in datalist]
        var = (sum(tmp_for_var)/len(dataMatrix))**0.5
        mean_list.append(mean)
        var_list.append(var)
    return mean_list, var_list

mean, var = mean_var(train_data)

def norm(dataMatrix, mean, var):
    #norm A_train
    for i in range(len(dataMatrix)):
        for j in range(len(mean)):

            if dataMatrix[i][j] != 0:
                try:
                    dataMatrix[i][j] = (dataMatrix[i][j] - mean[j])/var[j]
                except ZeroDivisionError:
                    dataMatrix[i][j]=0.
    return dataMatrix

norm_train = norm(train_data, mean, var)

def label_list(dataMatrix):
# get A_train label list
    train_label = []
    for i in range(len(dataMatrix)):
        train_label.append(dataMatrix[i][-1])
    label1 = [i for i in range(len(train_label)) if train_label[i]==1]
    label0 = [i for i in range(len(train_label)) if train_label[i]==0]
    return label1, label0

label1, label0 = label_list(norm_train)

subnum1 = len(label1) // 5
subnum0 = len(label0) // 5

subset1_1=[]
subset2_1=[]
subset3_1=[]
subset4_1=[]
subset5_1=[]

subset1_0=[]
subset2_0=[]
subset3_0=[]
subset4_0=[]
subset5_0=[]

for i in range(subnum1):
    subset1_1.append(norm_train[label1[i]])
    subset2_1.append(norm_train[label1[i + subnum1*1]])
    subset3_1.append(norm_train[label1[i + subnum1*2]])
    subset4_1.append(norm_train[label1[i + subnum1*3]])
for i in range(4*subnum1,len(label1)):
    subset5_1.append(norm_train[label1[i]])

for i in range(subnum0):
    subset1_0.append(norm_train[label0[i]])
    subset2_0.append(norm_train[label0[i + subnum0*1]])
    subset3_0.append(norm_train[label0[i + subnum0*2]])
    subset4_0.append(norm_train[label0[i + subnum0*3]])
for i in range(4*subnum0,len(label0)):
    subset5_0.append(norm_train[label0[i]])



val1 = subset1_0 + subset1_1
train1 = subset2_0 + subset3_0 + subset4_0 + subset5_0 + subset2_1 + subset3_1 + subset4_1 + subset5_1

val2 = subset2_0 + subset2_1
train2 = subset1_0 + subset3_0 + subset4_0 + subset5_0 + subset1_1 + subset3_1 + subset4_1 + subset5_1

val3 = subset3_0 + subset3_1
train3 = subset2_0 + subset1_0 + subset4_0 + subset5_0 + subset2_1 + subset1_1 + subset4_1 + subset5_1

val4 = subset4_0 + subset4_1
train4 = subset2_0 + subset3_0 + subset1_0 + subset5_0 + subset2_1 + subset3_1 + subset1_1 + subset5_1

val5 = subset5_0 + subset5_1
train5 = subset2_0 + subset3_0 + subset4_0 + subset1_0 + subset2_1 + subset3_1 + subset4_1 + subset1_1

np.save(save_path+'val1.npy',val1)
np.save(save_path+'val2.npy',val2)
np.save(save_path+'val3.npy',val3)
np.save(save_path+'val4.npy',val4)
np.save(save_path+'val5.npy',val5)

np.save(save_path+'train1.npy',train1)
np.save(save_path+'train2.npy',train2)
np.save(save_path+'train3.npy',train3)
np.save(save_path+'train4.npy',train4)
np.save(save_path+'train5.npy',train5)
