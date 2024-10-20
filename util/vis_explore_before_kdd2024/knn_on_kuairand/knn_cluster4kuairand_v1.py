import pickle
import torch
import os
import pandas as pd
from vis import svd_reduction_and_visualized_with_color, just_svd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import seaborn as sns
from collections import Counter

from src.core.configs import get_training_data, get_features
from environments.KuaiRec.env.KuaiEnv import KuaiEnv
import argparse
import random
from src.core.user_model_ensemble import EnsembleModel

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

with open('/home/s4715423/DORL-codes/saved_models/KuaiRand-v0/DeepFM/matsPre/[pointneg]_matPre.pickle', 'rb') as f:
    matpre = pickle.load(f)

## Don YZ: train data are in two csvs, train_processed corresponds to the first one,
## which means if we need to use both csvs, we need to pre-process the second one.
## In preprocessing_kuairand.py, we can find that the pro-processing focuses on duration, watch_ratio and watch_ratio_normed

## 这一版与上一版的区别在于：这版没用后一部分训练数据，以上一版为准

ground_truth_path = 'environments/KuaiRand_Pure/MF_results_GT/kuairand_is_click.csv'
train_part1_path = 'environments/KuaiRand_Pure/data/train_processed.csv'
# train_part2_path = 'environments/KuaiRand_Pure/data/log_standard_4_22_to_5_08_pure.csv'

df_gt = pd.read_csv(ground_truth_path, header=0, usecols=['user_id', 'item_id', 'value'])
df_gt.rename(columns={"value":"is_click"}, inplace=True)
df_train1 = pd.read_csv(train_part1_path, header=0, usecols=['user_id', 'item_id', 'is_click'])
# df_train2 = pd.read_csv(train_part2_path, header=0, usecols=['user_id', 'video_id', 'is_click'])
# df_train2.rename(columns={"video_id":"item_id"}, inplace=True)

df_train = df_train1

def load_mat(df_small):
    df_small.loc[df_small['is_click'] > 1, 'is_click'] = 1

    lbe_item = LabelEncoder()
    lbe_item.fit(df_small['item_id'].unique())

    lbe_user = LabelEncoder()
    lbe_user.fit(df_small['user_id'].unique())

    mat = csr_matrix( # Compressed Sparse Row matrix
        (df_small['is_click'],
            (lbe_user.transform(df_small['user_id']), lbe_item.transform(df_small['item_id']))),
        shape=(df_small['user_id'].nunique(), df_small['item_id'].nunique())).toarray() # mat of user*item

    mat[np.isnan(mat)] = df_small['is_click'].mean() # 用均值填补缺失值
    mat[np.isinf(mat)] = df_small['is_click'].mean()

    return mat, lbe_user, lbe_item

gt_mat, lbe_user, lbe_item = load_mat(df_gt)
train_mat = csr_matrix(
    (df_train['is_click'], (lbe_user.transform(df_train['user_id']), lbe_item.transform(df_train['item_id']))), \
        shape=(df_gt['user_id'].nunique(), df_gt['item_id'].nunique())).toarray()

diff_gt_pre = cosine_similarity(gt_mat, matpre)
diff_gt_pre_lst = np.diag(diff_gt_pre)

diff_gt_data = cosine_similarity(gt_mat, train_mat)
diff_gt_data_lst = np.diag(diff_gt_data)

def find_topK_to_compare(lst1, lst2, k, feature1, feature2):
    cross_mat = cosine_similarity(lst1, lst2)
    topk_indices = np.argsort(cross_mat, axis=1)[:, -k:] # obtain local indices
    diff_lst = []
    for i in range(len(topk_indices)):
        sim_users = topk_indices[i,:]
        features = feature1[sim_users,:]
        agg_feature = features.mean(0)
        gt_feature = feature2[i]
        diff_lst.append(1-distance.cosine(agg_feature, gt_feature))
    return diff_lst

diff_train_lst1 = find_topK_to_compare(train_mat, train_mat, 1, train_mat, gt_mat)
diff_train_lst3 = find_topK_to_compare(train_mat, train_mat, 3, train_mat, gt_mat)
diff_train_lst5 = find_topK_to_compare(train_mat, train_mat, 5, train_mat, gt_mat)
diff_train_lst7 = find_topK_to_compare(train_mat, train_mat, 7, train_mat, gt_mat)
diff_train_lst9 = find_topK_to_compare(train_mat, train_mat, 9, train_mat, gt_mat)
# diff_train_lst10000 = find_topK_to_compare(train_mat, train_mat, 10000, train_mat, gt_mat)

print(1)