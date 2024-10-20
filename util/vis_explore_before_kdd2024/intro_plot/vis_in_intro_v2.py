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

## Don YZ: the visualization used in the introduction of kdd submission

with open('/home/s4715423/DORL-codes/saved_models/KuaiEnv-v0/DeepFM/matsPre/[pointneg]_matPre.pickle', 'rb') as f:
    matpre = pickle.load(f)

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
# test_feat = torch.tensor(big_matrix[np.ix_(test_user_id, test_item_id)])

def find_topK_to_compare(lst1, lst2, k, feature1, feature2, clip):
    cross_mat = cosine_similarity(lst1, lst2)
    topk_indices = np.argsort(cross_mat, axis=1)[:, -k:] # obtain local indices
    pred_reward = np.zeros_like(feature2, dtype=float)
    diff_lst = []
    for i in range(len(topk_indices)):
        sim_users = topk_indices[i,:]
        features = feature1[sim_users,:]
        agg_feature = features.mean(0)
        pred_reward[i,:] = agg_feature
        gt_feature = feature2[i]
        diff_lst.append(1-distance.cosine(agg_feature, gt_feature))
    constrained_pred_reward = np.minimum(pred_reward, clip)
    return diff_lst, constrained_pred_reward

diff_lst_his_emb, pred_reward = find_topK_to_compare(block3_his_emb, block1_his_emb, 5, block2_feat, test_matrix, 5)

diff_matpre_gt = cosine_similarity(matpre, test_matrix)
diff_lst_matpre_gt = np.diag(diff_matpre_gt)

return_world_model_reward, return_interaction_history = np.round(20.4942/36.747,2), np.round(32.4232/36.747,2)

def calculate_stats(data):
    mean = np.mean(data)
    std = np.std(data)
    return mean, std

# Calculate stats for each set
mean2, std2 = calculate_stats(diff_lst_his_emb)
mean1, std1 = calculate_stats(diff_lst_matpre_gt)

bin_kuairec = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
# plot_binned_matrices(np.abs(test_matrix-matpre), np.abs(test_matrix-pred_reward), test_matrix, bin_kuairec)

dorl_sum_reward, roler_sim_reward = np.round(20.4942/36.747,2), np.round(33.2457/36.747,2)

def plot_combined_binned_matrices(matrix1, matrix2, matrix3, bins):
    """
    Draws a single bar plot for both matrix1 and matrix2 based on binning by matrix3.
    Bars for matrix1 and matrix2 are shown side-by-side for each bin for comparison.
    """
    if not (matrix1.shape == matrix2.shape == matrix3.shape):
        raise ValueError("All matrices must have the same shape.")

    # Flatten matrices
    matrix1_flat = matrix1.flatten()
    matrix2_flat = matrix2.flatten()
    matrix3_flat = matrix3.flatten()

    # Initialize lists to store averages
    averages_m1 = []
    averages_m2 = []

    # Calculate averages for each bin
    for i in range(len(bins) - 1):
        indices = np.where((matrix3_flat >= bins[i]) & (matrix3_flat < bins[i+1]))[0]
        
        avg_m1 = np.mean(matrix1_flat[indices]) if indices.size > 0 else 0
        avg_m2 = np.mean(matrix2_flat[indices]) if indices.size > 0 else 0
        
        averages_m1.append(avg_m1)
        averages_m2.append(avg_m2)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create an index for each bin group and width of each bar
    ind = np.arange(len(bins) - 1)  # the x locations for the groups
    width = 0.35       # the width of the bars

    # # Plot bars
    colors = ['#ef8a62', '#67a9cf']
    bars1 = ax.bar(ind - width/2, averages_m1, width, label='Prediction Error of the World Model', color=colors[0], alpha=0.6)
    bars2 = ax.bar(ind + width/2, averages_m2, width, label='Prediction Error of ROLeR', color=colors[1], alpha=0.6)

    # Labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Ground Truth Reward Intervals', fontsize=18)
    ax.set_ylabel('Average Prediction Error', fontsize=18)
    ax.set_xticks(ind)
    ax.set_xticklabels([f"{(bins[i] + bins[i+1]) / 2:.2f}" for i in range(len(bins)-1)])
    ax.legend(labels=["Prediction Error of the World Model; Relative Cumulative Reward: "+str(dorl_sum_reward), "Prediction Error of ROLeR; Relative Cumulative Reward: " + str(roler_sim_reward)], fontsize=14)

    plt.tight_layout()
    plt.savefig("intro.pdf", bbox_inches='tight')
    plt.show()
    plt.close()

plot_combined_binned_matrices(np.abs(test_matrix-matpre), np.abs(test_matrix-pred_reward), test_matrix, bin_kuairec)

print(1)