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

## Don YZ: investigate "the uncertainty-k relation" calculated by knn on KuaiRec

with open('/home/s4715423/DORL-codes/saved_models/KuaiEnv-v0/DeepFM/matsPre/[pointneg]_matPre.pickle', 'rb') as f:
    matpre = pickle.load(f)

## 这份代码在检查时需要注意big_matrix切分之后的各种id对应关系
## unlike in verion 1 directly mean, here we try weighted mean 

def load_mat(file):
    df_small = pd.read_csv(file, header=0, usecols=['user_id', 'item_id', 'watch_ratio'])
    # df_small['watch_ratio'][df_small['watch_ratio'] > 5] = 5
    df_small.loc[df_small['watch_ratio'] > 5, 'watch_ratio'] = 5

    lbe_item = LabelEncoder()
    lbe_item.fit(df_small['item_id'].unique())

    lbe_user = LabelEncoder()
    lbe_user.fit(df_small['user_id'].unique())

    mat = csr_matrix( # Compressed Sparse Row matrix
        (df_small['watch_ratio'],
            (lbe_user.transform(df_small['user_id']), lbe_item.transform(df_small['item_id']))),
        shape=(df_small['user_id'].nunique(), df_small['item_id'].nunique())).toarray() # mat of user*item

    mat[np.isnan(mat)] = df_small['watch_ratio'].mean() # 用均值填补缺失值
    mat[np.isinf(mat)] = df_small['watch_ratio'].mean()

    return mat, lbe_user, lbe_item

## train and test matrices
file_big = 'environments/KuaiRec/data/big_matrix_processed.csv'
file_small = "environments/KuaiRec/data/small_matrix_processed.csv"

big_matrix, big_lbe_user, big_lbe_item = load_mat(file_big)
test_matrix, test_lbe_user, test_lbe_item = load_mat(file_small)

user_embedding = torch.load('/home/s4715423/DORL-codes/saved_models/KuaiEnv-v0/DeepFM/embeddings/[pointneg]_emb_user_M0.pt')
embedding_layer = torch.nn.Embedding.from_pretrained(user_embedding, freeze=True)

## obtain train and test
user_num, item_num = big_matrix.shape
big_user_id, big_item_id = big_lbe_user.classes_, big_lbe_item.classes_
test_user_id, test_item_id = test_lbe_user.classes_, test_lbe_item.classes_
## the following means train only
train_user_id, train_item_id = np.setdiff1d(big_user_id, test_user_id), np.setdiff1d(big_item_id, test_item_id)

"""
 block1 | block2  
--------|--------
 block3 | test    
"""

block3_user_id_emb = embedding_layer(torch.tensor(test_user_id))
block12_user_id_emb = embedding_layer(torch.tensor(train_user_id))

block3_his_emb = torch.tensor(big_matrix[np.ix_(test_user_id, train_item_id)])
block1_his_emb = torch.tensor(big_matrix[np.ix_(train_user_id, train_item_id)])

## concat
block1_cat = torch.cat((block12_user_id_emb, block1_his_emb), 1)
block3_cat = torch.cat((block3_user_id_emb, block3_his_emb), 1)

## fetch features
block2_feat = torch.tensor(big_matrix[np.ix_(train_user_id, test_item_id)])

def find_topK_to_compare(lst1, lst2, k, feature1, feature2):
    cross_mat = cosine_similarity(lst1, lst2)
    topk_indices = np.argsort(cross_mat, axis=1)[:, -k:] # obtain local indices
    topk_values = np.sort(cross_mat, axis=1)[:, -k:]
    diff_lst = []
    uncertainty_lst = []
    uncertainty = np.zeros_like(feature2, dtype=float)
    for i in range(len(topk_indices)):
        sim_users = topk_indices[i,:]
        features = feature1[sim_users,:]
        # agg_feature = features.mean(0)
        uncertainty[i,:] = features.var(0)
        uncertainty_lst.append(features.var(0).mean())
    return uncertainty_lst, uncertainty

uncertainty_lst3, uncertainty3 = find_topK_to_compare(block3_his_emb, block1_his_emb, 3, block2_feat, test_matrix)
uncertainty_lst5, uncertainty5 = find_topK_to_compare(block3_his_emb, block1_his_emb, 5, block2_feat, test_matrix)
uncertainty_lst10, uncertainty10 = find_topK_to_compare(block3_his_emb, block1_his_emb, 10, block2_feat, test_matrix)
uncertainty_lst15, uncertainty15 = find_topK_to_compare(block3_his_emb, block1_his_emb, 15, block2_feat, test_matrix)
uncertainty_lst20, uncertainty20 = find_topK_to_compare(block3_his_emb, block1_his_emb, 20, block2_feat, test_matrix)
uncertainty_lst30, uncertainty30 = find_topK_to_compare(block3_his_emb, block1_his_emb, 30, block2_feat, test_matrix)
uncertainty_lst40, uncertainty40 = find_topK_to_compare(block3_his_emb, block1_his_emb, 40, block2_feat, test_matrix)
uncertainty_lst50, uncertainty50 = find_topK_to_compare(block3_his_emb, block1_his_emb, 50, block2_feat, test_matrix)

print(1)

