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
## used in KDD2024Feb Rebuttal.

def load_mat_kuairec(file):
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

def knn_based_pred_reward(env, kr, ku):
    if env == 'KuaiEnv-v0':
        ## train and test matrices
        file_big = 'environments/KuaiRec/data/big_matrix_processed.csv'
        file_small = "environments/KuaiRec/data/small_matrix_processed.csv"

        big_matrix, big_lbe_user, big_lbe_item = load_mat_kuairec(file_big)
        test_matrix, test_lbe_user, test_lbe_item = load_mat_kuairec(file_small)

        # user_embedding = torch.load('/home/s4715423/DORL-codes/saved_models/KuaiEnv-v0/DeepFM/embeddings/[pointneg]_emb_user_M0.pt')
        user_embedding = torch.load('./saved_models/KuaiEnv-v0/DeepFM/embeddings/[pointneg]_emb_user_M0.pt')
        embedding_layer = torch.nn.Embedding.from_pretrained(user_embedding, freeze=True)

        ## obtain train and test
        user_num, item_num = big_matrix.shape
        big_user_id, big_item_id = big_lbe_user.classes_, big_lbe_item.classes_
        test_user_id, test_item_id = test_lbe_user.classes_, test_lbe_item.classes_
        ## the following means train only
        train_user_id, train_item_id = np.setdiff1d(big_user_id, test_user_id), np.setdiff1d(big_item_id, test_item_id)

        """
        block1 | block2  
        -------|--------
        block3 | test    
        """

        block3_his_emb = torch.tensor(big_matrix[np.ix_(test_user_id, train_item_id)])
        block1_his_emb = torch.tensor(big_matrix[np.ix_(train_user_id, train_item_id)])

        ## fetch features
        block2_feat = torch.tensor(big_matrix[np.ix_(train_user_id, test_item_id)])

        pred_reward = np.zeros_like(test_matrix, dtype=float)
        uncertain_var = np.zeros_like(test_matrix, dtype=float)
        cross_mat = cosine_similarity(block3_his_emb, block1_his_emb)

        topkr_indices = np.argsort(cross_mat, axis=1)[:, -kr:]
        topku_indices = np.argsort(cross_mat, axis=1)[:, -ku:]

        for i in range(len(topkr_indices)):
            sim_users_kr = topkr_indices[i,:]
            sim_users_ku = topku_indices[i,:]

            rewards = block2_feat[sim_users_kr,:]
            pred_reward[i] = rewards.mean(0)

            uncertain = block2_feat[sim_users_ku,:]
            uncertain_var[i] = uncertain.var(0)

        return pred_reward, uncertain_var

    elif env == 'KuaiRand-v0':
        ground_truth_path = 'environments/KuaiRand_Pure/MF_results_GT/kuairand_is_click.csv'
        train_part1_path = 'environments/KuaiRand_Pure/data/train_processed.csv'
        train_part2_path = 'environments/KuaiRand_Pure/data/log_standard_4_22_to_5_08_pure.csv'
        # with open('/home/s4715423/DORL-codes/saved_models/KuaiRand-v0/DeepFM/matsPre/[pointneg]_matPre.pickle', 'rb') as f:
        with open('./saved_models/KuaiRand-v0/DeepFM/matsPre/[pointneg]_matPre.pickle', 'rb') as f:
            matpre = pickle.load(f)

        df_gt = pd.read_csv(ground_truth_path, header=0, usecols=['user_id', 'item_id', 'value'])
        df_gt.rename(columns={"value":"is_click"}, inplace=True)
        df_train1 = pd.read_csv(train_part1_path, header=0, usecols=['user_id', 'item_id', 'is_click'])
        df_train2 = pd.read_csv(train_part2_path, header=0, usecols=['user_id', 'video_id', 'is_click'])
        df_train2.rename(columns={"video_id":"item_id"}, inplace=True)

        df_train = pd.concat([df_train1, df_train2], ignore_index=True)

        gt_mat, lbe_user, lbe_item = load_mat_kuairand(df_gt)
        train_mat = csr_matrix(
            (df_train['is_click'], (lbe_user.transform(df_train['user_id']), lbe_item.transform(df_train['item_id']))), \
                shape=(df_gt['user_id'].nunique(), df_gt['item_id'].nunique())).toarray()

        ## user embedding
        # user_embedding = torch.load('/home/s4715423/DORL-codes/saved_models/KuaiRand-v0/DeepFM/embeddings/[pointneg]_emb_user_M0.pt')
        user_embedding = torch.load('./saved_models/KuaiRand-v0/DeepFM/embeddings/[pointneg]_emb_user_M0.pt')
        embedding_layer = torch.nn.Embedding.from_pretrained(user_embedding, freeze=True)
        user_embeddings = embedding_layer(torch.tensor(lbe_user.classes_))
        
        pred_reward = np.zeros_like(train_mat, dtype=float)
        uncertain_var = np.zeros_like(train_mat, dtype=float)

        ## matpre as feature
        pred_cross_mat = cosine_similarity(matpre, matpre)

        topkr_indices = np.argsort(pred_cross_mat, axis=1)[:, -kr:]
        topku_indices = np.argsort(pred_cross_mat, axis=1)[:, -ku:]

        for i in range(len(topkr_indices)):
            sim_users_kr = topkr_indices[i,:]
            sim_users_ku = topku_indices[i,:]

            rewards = matpre[sim_users_kr,:]
            pred_reward[i] = rewards.mean(0)

            uncertain = matpre[sim_users_ku,:]
            uncertain_var[i] = uncertain.var(0)

        return pred_reward, uncertain_var
    
    elif env == 'CoatEnv-v0':
        with open('./saved_models/CoatEnv-v0/DeepFM/matsPre/[pointneg]_matPre.pickle', 'rb') as f:
            matpre = pickle.load(f)

        ## GT for observation only
        filename_GT = 'environments/RL4Rec/data/coat_pseudoGT_ratingM.ascii'
        mat = pd.read_csv(filename_GT, sep="\s+", header=None, dtype=str).to_numpy(dtype=int)

        pred_reward = np.zeros_like(matpre, dtype=float)
        uncertain_var = np.zeros_like(matpre, dtype=float)

        ## matpre as feature
        pred_cross_mat = cosine_similarity(matpre, matpre)

        topkr_indices = np.argsort(pred_cross_mat, axis=1)[:, -kr:]
        topku_indices = np.argsort(pred_cross_mat, axis=1)[:, -ku:]

        for i in range(len(topkr_indices)):
            sim_users_kr = topkr_indices[i,:]
            sim_users_ku = topku_indices[i,:]

            rewards = matpre[sim_users_kr,:]
            pred_reward[i] = rewards.mean(0)

            uncertain = matpre[sim_users_ku,:]
            uncertain_var[i] = uncertain.var(0)

        return pred_reward, uncertain_var
    
    else:
        with open('./saved_models/YahooEnv-v0/DeepFM/matsPre/[pointneg]_matPre.pickle', 'rb') as f:
            matpre = pickle.load(f)

        ## GT for observation only
        filename_GT = 'environments/RL4Rec/data/yahoo_pseudoGT_ratingM.ascii'
        mat = pd.read_csv(filename_GT, sep="\s+", header=None, dtype=str).to_numpy(dtype=int)
        mat = mat[:5400,:]

        pred_reward = np.zeros_like(matpre, dtype=float)
        uncertain_var = np.zeros_like(matpre, dtype=float)

        ## matpre as feature
        pred_cross_mat = cosine_similarity(matpre, matpre)

        topkr_indices = np.argsort(pred_cross_mat, axis=1)[:, -kr:]
        topku_indices = np.argsort(pred_cross_mat, axis=1)[:, -ku:]

        for i in range(len(topkr_indices)):
            sim_users_kr = topkr_indices[i,:]
            sim_users_ku = topku_indices[i,:]

            rewards = matpre[sim_users_kr,:]
            pred_reward[i] = rewards.mean(0)

            uncertain = matpre[sim_users_ku,:]
            uncertain_var[i] = uncertain.var(0)

        return pred_reward, uncertain_var


# print(1)
if __name__ == '__main__':
    # rand_r_mat, rand_var_mat = knn_based_pred_reward("KuaiRand-v0")
    # rec_r_mat, rec_var_mat = knn_based_pred_reward("KuaiEnv-v0")
    # rand_r_mat, rand_var_mat = knn_based_pred_reward("CoatEnv-v0", 30, 30)
    rand_r_mat, rand_var_mat = knn_based_pred_reward("YahooEnv-v0", 50, 50)