import numpy as np
from sklearn.svm import SVC
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import numpy
import csv
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
import pandas as pd
from sklearn import preprocessing

def get_k_cross_validation_index(num_x, k=5, random_state=None):
    """
    @param:
        num_x: The size of dataset.
        k: THe k of k-th cross validation.
        random_state: The defaulting setting is None.
    """
    if random_state:
        np.random.seed(random_state)
    temp_rand_idx = np.random.permutation(num_x).tolist()

    temp_num_fold = num_x // k

    ret_tr_idx, ret_te_idx = {}, {}
    for i in range(k):
        temp_start_idx = i * temp_num_fold
        temp_end_idx = temp_start_idx + temp_num_fold
        ret_tr_idx[i] = temp_rand_idx[:temp_start_idx] + temp_rand_idx[temp_end_idx:]
        ret_te_idx[i] = temp_rand_idx[temp_start_idx: temp_end_idx]
    return ret_tr_idx, ret_te_idx



all = np.array(pd.read_csv('data/dataset/all.csv'))
print(all.shape)


# temp_num_x = len(train_labels)
# temp_k = 5
# temp_tr_idx, temp_te_idx = get_k_cross_validation_index(temp_num_x, temp_k)
# for i in range(temp_k):
#     clf = XGBClassifier()
#
#     trainfeatures = train_features[temp_tr_idx[i]]
#     trainlabels = train_labels[temp_tr_idx[i]]
#
#     clf.fit(trainfeatures, trainlabels)
#
#     # T is correct label
#     # S is predict score
#     # Y is predict label
#     file_AUCs = 'data/result/XGB/XGB' + str(i) + '--AUCs--new_labels_0_10.txt'
#     AUCs = ('AUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA')
#     with open(file_AUCs, 'w') as f:
#         f.write(AUCs + '\n')
#
#     testfeatures = train_features[temp_te_idx[i]]
#     testlabels = train_labels[temp_te_idx[i]]
#
#
#
#     S = numpy.array(list(map(lambda x: x[1], clf.predict_proba(testfeatures))))
#     T = testlabels
#     Y = clf.predict(testfeatures)
#     # T is correct label
#     # S is predict score
#     # Y is predict label
#
#     AUC = roc_auc_score(T, S)
#     precision, recall, threshold = metrics.precision_recall_curve(T, S)
#     PR_AUC = metrics.auc(recall, precision)
#     BACC = balanced_accuracy_score(T, Y)
#     tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
#     TPR = tp / (tp + fn)
#     PREC = precision_score(T, Y)
#     ACC = accuracy_score(T, Y)
#     KAPPA = cohen_kappa_score(T, Y)
#
#     def save_AUCs(AUCs, filename):
#         with open(filename, 'a') as f:
#             f.write('\t'.join(map(str, AUCs)) + '\n')
#
#     # save data
#
#     AUCs = [AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA]
#     save_AUCs(AUCs, file_AUCs)
#
#     independent_num = []
#     independent_num.append(T)
#     independent_num.append(Y)
#     independent_num.append(S)
#     txtDF = pd.DataFrame(data=independent_num)
#     txtDF.to_csv('data/result/XGB/XGB' + str(i) + '--ROC--new_labels_0_10.csv', index=False, header=False)