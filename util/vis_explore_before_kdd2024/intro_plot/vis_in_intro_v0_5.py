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

ground_truth_path = 'environments/KuaiRand_Pure/MF_results_GT/kuairand_is_click.csv'
df_gt = pd.read_csv(ground_truth_path, header=0, usecols=['user_id', 'item_id', 'value'])
df_gt.rename(columns={"value":"is_click"}, inplace=True)
setgtitem = df_gt["item_id"].unique()
setgtuser = df_gt["user_id"].unique()

def load_mat_rand(df_small):
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

gt_mat, lbe_user, lbe_item = load_mat_rand(df_gt)

def find_topK_to_compare(lst1, lst2, k, feature1, feature2, clip):
    cross_mat = cosine_similarity(lst1, lst2)
    topk_indices = np.argsort(cross_mat, axis=1)[:, -k:] # obtain local indices
    pred_reward = np.zeros_like(feature2, dtype=float)
    diff_lst = []
    for i in range(len(topk_indices)):
        sim_users = topk_indices[i,:]
        features = feature1[sim_users,:]
        agg_feature = features.mean(0)
        # pred_reward[i,:] = np.minimum(agg_feature, clip)
        pred_reward[i,:] = agg_feature
        gt_feature = feature2[i]
        diff_lst.append(1-distance.cosine(agg_feature, gt_feature))
    return diff_lst, pred_reward

diff_lst_his_emb, pred_kuairand = find_topK_to_compare(matpre, matpre, 1000, matpre, gt_mat, 1)

diff_matpre_gt = cosine_similarity(matpre, gt_mat)
diff_lst_matpre_gt = np.diag(diff_matpre_gt)

return_world_model_reward, return_interaction_history = np.round(11.85/13.902,2), np.round(13.1833/13.902,2)

def calculate_stats(data):
    mean = np.mean(data)
    std = np.std(data)
    return mean, std

# Calculate stats for each set
mean2, std2 = calculate_stats(diff_lst_his_emb)
mean4, std4 = calculate_stats(diff_lst_matpre_gt)

fig, ax1 = plt.subplots()
ax1.boxplot([diff_lst_matpre_gt, diff_lst_his_emb], labels=['World Model Reward', 'Interaction History'])
ax1.set_ylabel("Cosine Similarity")
ax1.set_xlabel("Groups")
ax1.set_title("Cosine Similarity and Reward/Interaction History")

# Add text for mean and std - Adjust positions as necessary
ax1.text(1, mean4 + std4, f'Mean: {mean4:.2f}\nSTD: {std4:.2f}', ha='center')
ax1.text(2, mean2 + std2, f'Mean: {mean2:.2f}\nSTD: {std2:.2f}', ha='center')


# Create the secondary y-axis for the line charts
ax2 = ax1.twinx()
ax2.plot([1, 2], [return_world_model_reward, return_interaction_history], 'r-')
ax2.set_ylabel("Reward/Interaction History", color='r')
ax2.set_ylim(0,1)
for tl in ax2.get_yticklabels():
    tl.set_color('r')

plt.ylabel("Values")
plt.xlabel("Groups")
plt.show()
plt.close()

print(1)