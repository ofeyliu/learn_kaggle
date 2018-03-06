# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 16:12:50 2017

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
save_path = 'result/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


def convert_to_float(datapath):
    train=[]
    f = open(datapath,'r')
    num = 10000
    for i in range(num):
        line = f.readline()
        line = line.strip('\n')
        line=line.split()
        line.append(line[6])
        line.remove(line[6])
        train.append(line)
    train.remove(train[0])

# convert items to float list
    train_new = []
    for i in range(len(train)):
        item = []
        train[i] = train[i][6:]
        for j in range(len(train[i])):
            if train[i][j]=='NULL':
                item.append(-1.)
                item.append(1.)
            elif j !=len(train[0])-1:
                item.append(float(train[i][j]))
                item.append(0.)
            else :
                item.append(float(train[i][j]))
        train_new.append(item)
    return train_new, train
train_data, orginal = convert_to_float(path)

def mean_var(dataMatrix, orginalMatrix):
    mean_list = []
    var_list = []
    for i in range(len(orginalMatrix[0])-1):
        datalist = [x[i] for x in dataMatrix if x[2*i]!= -1.]
        mean = sum(datalist)/len(dataMatrix)
        tmp_for_var = [(x-mean)**2 for x in datalist]
        var = (sum(tmp_for_var)/len(dataMatrix))**0.5
        mean_list.append(mean)
        var_list.append(var)
    return mean_list, var_list

mean, var = mean_var(train_data, orginal)

def norm(dataMatrix, mean, var):
    #norm A_train

    for i in range(len(dataMatrix)):
        for j in range(len(mean)):
            try:
                dataMatrix[i][2*j] = (dataMatrix[i][2*j] - mean[j])/var[j]
            except ZeroDivisionError:
                dataMatrix[i][2*j]=0.
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


def save_txt(Matrix,name):
    fo = open(name+'txt', 'w')
    for line in Matrix:
        fo.write(str(line[-1]))
        fo.write('\t')
        for index,element in enumerate(line):
            if index == len(line):
                continue
            fo.write('%d:%d' % (index, line[index]))
            fo.write('\t')
        fo.write('\n')
save_txt(val1,'val1')
save_txt(val2,'val2')
save_txt(val3,'val3')
save_txt(val4,'val4')
save_txt(val5,'val5')
save_txt(train1,'train1')
save_txt(train2,'train2')
save_txt(train3,'train3')
save_txt(train4,'train4')
save_txt(train5,'train5')




'''
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
'''