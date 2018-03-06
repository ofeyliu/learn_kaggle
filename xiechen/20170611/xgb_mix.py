'''
@author: Chen Xu
@time: 2017/04/18
'''
import xgboost as xgb
import os
import numpy as np
import pickle
def get_dataset(path):
    B_train_dataset_list = []
    B_validate_dataset_list = []
    for i in range(5):
        B_train_dataset = np.load(path+'/data/'+str(i+1)+'/B_train.npy')
        B_validate_dataset = np.load(path+'/data/'+str(i+1)+'/B_val.npy')
        B_train_dataset_list.append(B_train_dataset)
        B_validate_dataset_list.append(B_validate_dataset)
    return B_train_dataset_list, B_validate_dataset_list

def train(B_train_datset, B_validate_dataset, parameters, num_round, output_model_file, output_param_file):
    B_train_data = B_train_datset[:, :-1]
    B_train_labels =B_train_datset[:, -1]
    B_trainDMatrix = xgb.DMatrix(B_train_data, label=B_train_labels)

    B_vali_data = B_validate_dataset[:, :-1]
    B_vali_labels = B_validate_dataset[:, -1]
    B_valiDMatrix = xgb.DMatrix(B_vali_data, label=B_vali_labels)
    evallist = [(B_valiDMatrix, 'test'), (B_trainDMatrix, 'train')]

    bst = xgb.train(parameters, B_trainDMatrix, num_round, evallist)
    bst.save_model(output_model_file)
    pickle.dump(parameters, open(output_param_file, 'w'))

if __name__ == '__main__':
    TRAIN_FLAG = 1
    PREDICTION_FLAG = 0

    path = os.getcwd()
    B_train_dataset_list, B_validate_dataset_list = get_dataset(path)
    B_test_dataset = np.load(path+'/data/B_test.npy')

    if TRAIN_FLAG==1:
        parameters = {'bst:max_depth': 3, 'bst:eta': 0.3, 'silent': 1, 'objective': 'binary:logistic', 'nthread': 4, 'eval_metric': 'auc',
                        'base_score': 0.1, 'lamabda': 1}
        # xgb.cv()
        num_rounds_list = [4, 4, 18, 18, 4]
        model_list = []
        for i in range(5):
            print 'training model '+str(i)
            output_model_file = path+'/data/model_mix_'+str(i)+'.model'
            output_param_file = path+'/data/params_mix_'+str(i)+'.pkl'
            train(B_train_dataset_list[i], B_validate_dataset_list[i], parameters, num_rounds_list[i], output_model_file, output_param_file)

    if PREDICTION_FLAG == 1:

        uid_list = []
        test_DMatrix = xgb.DMatrix(B_test_dataset)
        with open(path+'/data/B_test.csv', 'r') as f:
            lines = f.readlines()[1:]
            for i in range(len(lines)):
                uid_list.append(lines[i].strip().split(',')[0])

        train_model_list = []
        five_model_prediction = []
        for i in range(5):
            bst = xgb.Booster({'nthread': 4})
            model_path = path+'/data/'+'model_mix_'+str(i)+'.model'
            bst.load_model(model_path)
            train_model_list.append(bst)
        for i in range(5):
            print 'prediction model '+str(i)+'\n'
            five_model_prediction.append(list(train_model_list[i].predict(test_DMatrix)))
        five_model_prediction = np.array(five_model_prediction)
        average_prediction = np.mean(five_model_prediction, 0)

        with open(path+'/data/submit_mix.csv', 'w') as f:
            f.writelines('no,pred'+'\n')
            for i in range(len(uid_list)):
                f.writelines(uid_list[i]+','+str(average_prediction[i])+'\n')

