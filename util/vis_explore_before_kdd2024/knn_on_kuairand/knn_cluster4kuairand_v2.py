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

from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy.spatial import distance

## Don YZ: knn clustering investigation on KuaiRand (newest) 

with open('/home/s4715423/DORL-codes/saved_models/KuaiRand-v0/DeepFM/matsPre/[pointneg]_matPre.pickle', 'rb') as f:
    matpre = pickle.load(f)

user_embedding = torch.load('/home/s4715423/DORL-codes/saved_models/KuaiRand-v0/DeepFM/embeddings/[pointneg]_emb_user_M0.pt')
embedding_layer = torch.nn.Embedding.from_pretrained(user_embedding, freeze=True)

## train data is in two csvs, train_processed corresponds to the first one,
## which means if we need to use both csvs, we need to pre-process the second one.
## In preprocessing_kuairand.py, we can find that the pro-processing focuses on duration, watch_ratio and watch_ratio_normed

ground_truth_path = 'environments/KuaiRand_Pure/MF_results_GT/kuairand_is_click.csv'
train_part1_path = 'environments/KuaiRand_Pure/data/train_processed.csv'
train_part2_path = 'environments/KuaiRand_Pure/data/log_standard_4_22_to_5_08_pure.csv'

df_gt = pd.read_csv(ground_truth_path, header=0, usecols=['user_id', 'item_id', 'value'])
df_gt.rename(columns={"value":"is_click"}, inplace=True)
df_train1 = pd.read_csv(train_part1_path, header=0, usecols=['user_id', 'item_id', 'is_click'])
df_train2 = pd.read_csv(train_part2_path, header=0, usecols=['user_id', 'video_id', 'is_click'])
df_train2.rename(columns={"video_id":"item_id"}, inplace=True)

## some checking
set1item = df_train1["item_id"].unique()
set2item = df_train2["item_id"].unique()
setgtitem = df_gt["item_id"].unique()

set1user = df_train1["user_id"].unique()
set2user = df_train2["user_id"].unique()
setgtuser = df_gt["user_id"].unique()

df_train = pd.concat([df_train1, df_train2], ignore_index=True)

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

user_embeddings = embedding_layer(torch.tensor(lbe_user.classes_))
emb_plus_hist = torch.cat((user_embeddings, torch.tensor(train_mat)), 1)

## using cosine similarity
diff_gt_pre = cosine_similarity(gt_mat, matpre)
diff_gt_pre_lst = np.diag(diff_gt_pre)

diff_gt_data = cosine_similarity(gt_mat, train_mat)
diff_gt_data_lst = np.diag(diff_gt_data) # mean 0.10


def find_topK_to_compare_cos(lst1, lst2, k, feature1, feature2):
    cross_mat = cosine_similarity(lst1, lst2)
    topk_indices = np.argsort(cross_mat, axis=1)[:, -k:] # obtain local indices
    diff_lst = []
    for i in range(len(topk_indices)):
        sim_users = topk_indices[i,:]
        features = feature1[sim_users,:]
        # agg_feature = features.mean(0)
        # agg_feature = features.min(0)
        agg_feature = features.max(0)
        # agg_feature = features.sum(0)
        gt_feature = feature2[i]
        diff_lst.append(1-distance.cosine(agg_feature, gt_feature))
    return diff_lst

# diff_train_lst1 = find_topK_to_compare_cos(train_mat, train_mat, 1, train_mat, gt_mat)
# diff_train_lst3 = find_topK_to_compare_cos(train_mat, train_mat, 3, train_mat, gt_mat)
# diff_train_lst5 = find_topK_to_compare_cos(train_mat, train_mat, 5, train_mat, gt_mat) # np.mean()=0.142
# diff_train_lst7 = find_topK_to_compare_cos(train_mat, train_mat, 7, train_mat, gt_mat)
# diff_train_lst9 = find_topK_to_compare_cos(train_mat, train_mat, 9, train_mat, gt_mat)
# diff_train_lst100 = find_topK_to_compare_cos(train_mat, train_mat, 100, train_mat, gt_mat) # np.mean(diff_train_lst100(min))=0.9745
# diff_train_lst1000 = find_topK_to_compare_cos(train_mat, train_mat, 1000, train_mat, gt_mat) # np.mean(diff_train_lst1000(min))=0.9934
# diff_train_lst10000 = find_topK_to_compare_cos(train_mat, train_mat, 10000, train_mat, gt_mat)

# diff_train_lst1 = find_topK_to_compare_cos(user_embeddings, user_embeddings, 1, train_mat, gt_mat)
# diff_train_lst3 = find_topK_to_compare_cos(user_embeddings, user_embeddings, 3, train_mat, gt_mat)
# diff_train_lst5 = find_topK_to_compare_cos(user_embeddings, user_embeddings, 5, train_mat, gt_mat)
# diff_train_lst7 = find_topK_to_compare_cos(user_embeddings, user_embeddings, 7, train_mat, gt_mat)
# diff_train_lst9 = find_topK_to_compare_cos(user_embeddings, user_embeddings, 9, train_mat, gt_mat)
# diff_train_lst9 = find_topK_to_compare_cos(user_embeddings, user_embeddings, 10, train_mat, gt_mat) # np.max(200)->0.57;500->0.65;100->0.70

# diff_train_lst1 = find_topK_to_compare_cos(emb_plus_hist, emb_plus_hist, 1, train_mat, gt_mat)
# diff_train_lst3 = find_topK_to_compare_cos(emb_plus_hist, emb_plus_hist, 3, train_mat, gt_mat)
# diff_train_lst5 = find_topK_to_compare_cos(emb_plus_hist, emb_plus_hist, 5, train_mat, gt_mat)
# diff_train_lst7 = find_topK_to_compare_cos(emb_plus_hist, emb_plus_hist, 7, train_mat, gt_mat)
# diff_train_lst9 = find_topK_to_compare_cos(emb_plus_hist, emb_plus_hist, 9, train_mat, gt_mat)

# diff_train_lst1 = find_topK_to_compare_cos(matpre, matpre, 1, matpre, gt_mat) # 0.704
# diff_train_lst3 = find_topK_to_compare_cos(matpre, matpre, 3, matpre, gt_mat)
# diff_train_lst5 = find_topK_to_compare_cos(matpre, matpre, 5, matpre, gt_mat)
# diff_train_lst7 = find_topK_to_compare_cos(matpre, matpre, 7, matpre, gt_mat)
# diff_train_lst9 = find_topK_to_compare_cos(matpre, matpre, 9, matpre, gt_mat) # 0.712
# diff_train_lst100 = find_topK_to_compare_cos(matpre, matpre, 100, matpre, gt_mat) # 0.727
diff_train_lst1000 = find_topK_to_compare_cos(matpre, matpre, 1000, matpre, gt_mat) # 0.755

## repeat with l2 distance
diff_gt_pre = pairwise_distances(gt_mat, matpre, 'l2')
diff_gt_pre_lst = np.diag(diff_gt_pre)

diff_gt_data = pairwise_distances(gt_mat, train_mat, 'l2')
diff_gt_data_lst = np.diag(diff_gt_data)

def find_topK_to_compare_l2(lst1, lst2, k, feature1, feature2):
    cross_mat = pairwise_distances(lst1, lst2, 'l2')
    topk_indices = np.argsort(cross_mat, axis=1)[:, :k] # obtain local indices
    diff_lst = []
    for i in range(len(topk_indices)):
        sim_users = topk_indices[i,:]
        features = feature1[sim_users,:]
        agg_feature = features.mean(0)
        # agg_feature = features.min(0)
        # agg_feature = features.max(0)
        # agg_feature = features.sum(0)
        gt_feature = feature2[i]
        diff_lst.append(distance.euclidean(agg_feature, gt_feature))
    return diff_lst

# diff_train_lst1_l2 = find_topK_to_compare_l2(train_mat, train_mat, 1, train_mat, gt_mat)
# diff_train_lst3_l2 = find_topK_to_compare_l2(train_mat, train_mat, 3, train_mat, gt_mat)
# diff_train_lst5_l2 = find_topK_to_compare_l2(train_mat, train_mat, 5, train_mat, gt_mat)
# diff_train_lst7_l2 = find_topK_to_compare_l2(train_mat, train_mat, 7, train_mat, gt_mat)
# diff_train_lst9_l2 = find_topK_to_compare_l2(train_mat, train_mat, 9, train_mat, gt_mat)

diff_train_lst1_l2 = find_topK_to_compare_l2(user_embeddings, user_embeddings, 1, train_mat, gt_mat)
diff_train_lst3_l2 = find_topK_to_compare_l2(user_embeddings, user_embeddings, 3, train_mat, gt_mat)
diff_train_lst5_l2 = find_topK_to_compare_l2(user_embeddings, user_embeddings, 5, train_mat, gt_mat)
diff_train_lst7_l2 = find_topK_to_compare_l2(user_embeddings, user_embeddings, 7, train_mat, gt_mat)
diff_train_lst9_l2 = find_topK_to_compare_l2(user_embeddings, user_embeddings, 9, train_mat, gt_mat)

print(1)