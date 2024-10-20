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

## Don YZ: investigate "the uncertainty&k relation" calculated by knn on KuaiRand

def load_mat_kuairand(df_small):
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

ground_truth_path = 'environments/KuaiRand_Pure/MF_results_GT/kuairand_is_click.csv'
with open('/home/s4715423/DORL-codes/saved_models/KuaiRand-v0/DeepFM/matsPre/[pointneg]_matPre.pickle', 'rb') as f:
    matpre = pickle.load(f)

df_gt = pd.read_csv(ground_truth_path, header=0, usecols=['user_id', 'item_id', 'value'])
df_gt.rename(columns={"value":"is_click"}, inplace=True)

gt_mat, lbe_user, lbe_item = load_mat_kuairand(df_gt)

def uncertain(matpre, k):
    pred_reward = np.zeros_like(matpre, dtype=float)
    uncertain_var = np.zeros_like(matpre, dtype=float)

    ## matpre as feature
    pred_cross_mat = cosine_similarity(matpre, matpre)

    topkr_indices = np.argsort(pred_cross_mat, axis=1)[:, -k:]
    topku_indices = np.argsort(pred_cross_mat, axis=1)[:, -k:]

    for i in range(len(topkr_indices)):
        sim_users_kr = topkr_indices[i,:]
        sim_users_ku = topku_indices[i,:]

        rewards = matpre[sim_users_kr,:]
        pred_reward[i] = rewards.mean(0)

        uncertain = matpre[sim_users_ku,:]
        uncertain_var[i] = uncertain.var(0)

    return pred_reward, uncertain_var

_, uncertainty25 = uncertain(matpre, 25)
# _, uncertainty50 = uncertain(matpre, 50)

_, uncertainty75 = uncertain(matpre, 75)
# _, uncertainty100 = uncertain(matpre, 100)

_, uncertainty125 = uncertain(matpre, 125)
# _, uncertainty150 = uncertain(matpre, 150)

_, uncertainty175 = uncertain(matpre, 175)
# _, uncertainty200 = uncertain(matpre, 200)

print(1)