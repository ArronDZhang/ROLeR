import pickle
import torch
import os
import pandas as pd
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

## 这份代码在检查时需要注意big_matrix切分之后的各种id对应关系

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

def knn_based_pred_reward():
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

    block3_his_emb = torch.tensor(big_matrix[np.ix_(test_user_id, train_item_id)])
    block1_his_emb = torch.tensor(big_matrix[np.ix_(train_user_id, train_item_id)])

    ## fetch features
    block2_feat = torch.tensor(big_matrix[np.ix_(train_user_id, test_item_id)])

    pred_reward = np.zeros_like(test_matrix, dtype=float)
    cross_mat = cosine_similarity(block3_his_emb, block1_his_emb)
    top5_indices = np.argsort(cross_mat, axis=1)[:, -5:] # obtain local indices
    for i in range(len(top5_indices)):
        sim_users = top5_indices[i,:]
        rewards = block2_feat[sim_users,:]
        pred_reward[i] = rewards.mean(0)

    return pred_reward

# print(1)
