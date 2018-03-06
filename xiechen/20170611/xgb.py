# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 08:56:04 2017

@author: li_yu
"""

import xgboost as xgb
import numpy as np
import os
from sklearn.decomposition import KernelPCA, PCA
import pickle
TRAIN_FLAG = 1
PREDICTION_FLAG = 1

def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label==1)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)

def pca_data(B_train_dataset, B_test_dataset):
    num_train = np.size(B_train_dataset, 0)
    X = np.concatenate((B_train_dataset[:, :-1], B_test_dataset), 0)
    kpca = PCA(n_components=0.9)
    X_kpca = kpca.fit_transform(X)

    return X_kpca[:num_train, :], X_kpca[num_train:, :]



if __name__ == '__main__':

    path = os.getcwd()
    B_train_dataset = np.load(path+'/data/B_train.npy')
    B_test_dataset = np.load(path+'/data/B_test.Qnpy')

    B_train_DMatrix = xgb.DMatrix(B_train_dataset[:, :-1], label=B_train_dataset[:, -1])
    B_test_DMatrix = xgb.DMatrix(B_test_dataset)



    if TRAIN_FLAG == 1:
        print('training process...')

        label = B_train_DMatrix.get_label()
        ratio = float(np.sum(label == 0)) / np.sum(label==1)

        parameters = {'bst:max_depth': 3, 'bst:eta': 0.3, 'silent': 1, 'objective': 'binary:logistic', 'nthread': 4, 'eval_metric': 'auc',
                            'base_score': 0.1, 'lamabda': 1, 'scale_pos_weight': ratio}
        num_rounds = 20
        res = xgb.cv(parameters, B_train_DMatrix, num_boost_round=5, nfold=5, callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                                                                                    xgb.callback.early_stop(20)])
        print(res)

        evallist = [(B_train_DMatrix, 'train')]
        bst = xgb.train(parameters, B_train_DMatrix, len(res), evallist)
        bst.save_model(path+'/data/model_final.model')
    if PREDICTION_FLAG == 1:
        print('testing process...')
        bst = xgb.Booster({'nthread': 4})
        bst.load_model(path+'/data/model_final.model')
        prediction = bst.predict(B_test_DMatrix)

        uid_list = []
        with open(path+'/data/B_test.csv', 'r') as f:
            lines = f.readlines()[1:]
            for i in range(len(lines)):
                uid_list.append(lines[i].strip().split(',')[0])

        with open(path+'/data/submit_final.csv', 'w') as f:
            f.writelines('no,pred'+'\n')
            for i in range(len(uid_list)):
                f.writelines(uid_list[i]+','+str(prediction[i])+'\n')