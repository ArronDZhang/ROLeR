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

## adapted from pre_utilv4.py for initialization testing
## 这份代码在检查时需要注意big_matrix切分之后的各种id对应关系

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

## for yahoo
def get_df_yahoo(filename):
    # read interaction
    df_data = pd.read_csv(filename, sep="\s+", header=None, names=["user_id", "item_id", "rating"])

    df_data["user_id"] -= 1
    df_data["item_id"] -= 1

    df_user = load_user_feat()
    df_item = load_item_feat()
    list_feat = None

    return df_data, df_user, df_item, list_feat

def load_user_feat():
    df_user = pd.DataFrame(np.arange(15400), columns=["user_id"])
    df_user.set_index("user_id", inplace=True)
    return df_user

def load_item_feat():
    df_item = pd.DataFrame(np.arange(1000), columns=["item_id"])
    df_item.set_index("item_id", inplace=True)
    return df_item

def ew_min(mat1, mat3): # calculate the element-wise min as distance metric
    ## block1 as mat1, block3 as mat3
    ew_dis = np.sum(mat1 - mat3, dim=0)
    return ew_dis

def knn_based_pred_reward(env, kr, ku, uncertain_type, init_data='offline', num_iter=1):
    if env == 'KuaiEnv-v0':
        ## train and test matrices
        file_big = 'environments/KuaiRec/data/big_matrix_processed.csv'
        file_small = "environments/KuaiRec/data/small_matrix_processed.csv"

        big_matrix, big_lbe_user, big_lbe_item = load_mat_kuairec(file_big)
        test_matrix, test_lbe_user, test_lbe_item = load_mat_kuairec(file_small)

        # user_embedding = torch.load('/home/s4715423/DORL-codes/saved_models/KuaiEnv-v0/DeepFM/embeddings/[pointneg]_emb_user_M0.pt')
        user_embedding = torch.load('./saved_models/KuaiEnv-v0/DeepFM/embeddings/[pointneg]_emb_user_M0.pt')
        embedding_layer = torch.nn.Embedding.from_pretrained(user_embedding, freeze=True)

        with open('./saved_models/KuaiEnv-v0/DeepFM/matsPre/[pointneg]_matPre.pickle', 'rb') as f:
            matpre = pickle.load(f)

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

        ## np.ix_(rows, cols)用于从矩阵中取指定的行和列
        block3_his_emb = torch.tensor(big_matrix[np.ix_(test_user_id, train_item_id)])
        block1_his_emb = torch.tensor(big_matrix[np.ix_(train_user_id, train_item_id)])

        ## fetch features
        block2_feat = torch.tensor(big_matrix[np.ix_(train_user_id, test_item_id)])

        pred_reward = np.zeros_like(test_matrix, dtype=float)
        uncertain_pen = np.zeros_like(test_matrix, dtype=float)
        cross_mat = cosine_similarity(block3_his_emb, block1_his_emb)

        topkr_indices = np.argsort(cross_mat, axis=1)[:, -kr:]
        topku_indices = np.argsort(cross_mat, axis=1)[:, -ku:]

        topku_sim = np.sort(cross_mat, axis=1)[:, -ku:]
        cos_dist_ku = 1-topku_sim
        num_test_user, num_test_item = test_matrix.shape

        for i in range(len(topkr_indices)):
            sim_users_kr = topkr_indices[i,:]
            sim_users_ku = topku_indices[i,:]

            rewards = block2_feat[sim_users_kr,:]
            pred_reward[i] = rewards.mean(0)

            ## the variance of neighbors as uncertainty [Feb2024KDD Setting]
            if uncertain_type == 'I-var':
                uncertain = block2_feat[sim_users_ku,:]
                uncertain_pen[i] = uncertain.var(0)

            ## attempt 1: the cos_dist. between the user and its neighbor in block1and3 (indicator factor)
            ## since there will be #neighbors distances, we can try the min, mean, and max
            elif uncertain_type == 'I-cos_min':
                uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].min()
            elif uncertain_type == 'I-cos_avg':
                uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].mean()
            elif uncertain_type == 'I-cos_max':
                uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].max()

            ## attempt 2: the cos_dist. between the user and its neighbor in block1and3 (indicator factor)
            ## we weight the predicted r with cos_similarity
            elif uncertain_type == "II-weight":
                uncertain_weights = torch.tensor(1-cos_dist_ku[i]).unsqueeze(1)
                pred_reward[i] = torch.sum(uncertain_weights * rewards, dim=0)
            elif uncertain_type == "II-weight-div":
                uncertain_weights = torch.tensor(1-cos_dist_ku[i]).unsqueeze(1)
                pred_reward[i] = torch.sum(uncertain_weights * rewards, dim=0)/kr
            elif uncertain_type == "II-weight-norm":
                uncertain_weights_norm = torch.tensor((1-cos_dist_ku[i])/(1-cos_dist_ku[i]).sum()).unsqueeze(1)
                pred_reward[i] = torch.sum(uncertain_weights_norm * rewards, dim=0)

            ## ablation for type 'II-weight-'; unreasonable --> abort
            elif uncertain_type == 'abla':
                uncertain_weights_norm = torch.tensor((1-cos_dist_ku[i])/(1-cos_dist_ku[i]).sum()).unsqueeze(1)
                uncertain_pen[i] = uncertain_weights_norm

        return pred_reward, uncertain_pen

    elif env == 'KuaiRand-v0': # KuaiRand-v0
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
        uncertain_pen = np.zeros_like(train_mat, dtype=float)

        if init_data == 'offline':
            pred_cross_mat = cosine_similarity(train_mat, train_mat)
            topkr_indices = np.argsort(pred_cross_mat, axis=1)[:, -kr:]
            topku_indices = np.argsort(pred_cross_mat, axis=1)[:, -ku:]

            topku_sim = np.sort(pred_cross_mat, axis=1)[:, -ku:]
            cos_dist_ku = 1-topku_sim+1e-9
            num_test_user, num_test_item = train_mat.shape

            for j in range(num_iter):
                for i in range(len(topkr_indices)):
                    sim_users_kr = topkr_indices[i,:]
                    sim_users_ku = topku_indices[i,:]

                    rewards = train_mat[sim_users_kr,:]
                    pred_reward[i] = rewards.mean(0)

                    ## Feb2024KDD Setting
                    if uncertain_type == 'I-var':
                        uncertain = train_mat[sim_users_ku,:]
                        uncertain_pen[i] = uncertain.var(0)

                    ## attempt 1: the cos_dist. between the user and its neighbor in train set
                    elif uncertain_type == 'I-cos_min':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].min()
                    elif uncertain_type == 'I-cos_avg':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].mean()
                    elif uncertain_type == 'I-cos_max':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].max()

                    ## attempt 2: the cos_dist. between the user and its neighbor in block1and3 (indicator factor)
                    ## we weight the predicted r with cos_similarity
                    elif uncertain_type == "II-weight":
                        uncertain_weights = torch.tensor(1-cos_dist_ku[i]).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights * rewards, dim=0)
                    elif uncertain_type == "II-weight-div":
                        uncertain_weights = torch.tensor(1-cos_dist_ku[i]).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights * rewards, dim=0)/kr
                    elif uncertain_type == "II-weight-norm":
                        uncertain_weights_norm = torch.tensor((1-cos_dist_ku[i])/(1-cos_dist_ku[i]).sum()).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights_norm * rewards, dim=0)

                pred_cross_mat = cosine_similarity(pred_reward, pred_reward)
                topkr_indices = np.argsort(pred_cross_mat, axis=1)[:, -kr:]
                topku_indices = np.argsort(pred_cross_mat, axis=1)[:, -ku:]
                topku_sim = np.sort(pred_cross_mat, axis=1)[:, -ku:]
                cos_dist_ku = 1-topku_sim+1e-9

        elif init_data == 'wm':
            pred_cross_mat = cosine_similarity(matpre, matpre)
            topkr_indices = np.argsort(pred_cross_mat, axis=1)[:, -kr:]
            topku_indices = np.argsort(pred_cross_mat, axis=1)[:, -ku:]

            topku_sim = np.sort(pred_cross_mat, axis=1)[:, -ku:]
            cos_dist_ku = 1-topku_sim+1e-9
            num_test_user, num_test_item = matpre.shape

            for j in range(num_iter):
                for i in range(len(topkr_indices)):
                    sim_users_kr = topkr_indices[i,:]
                    sim_users_ku = topku_indices[i,:]

                    rewards = matpre[sim_users_kr,:]
                    pred_reward[i] = rewards.mean(0)

                    ## Feb2024KDD Setting
                    if uncertain_type == 'I-var':
                        uncertain = matpre[sim_users_ku,:]
                        uncertain_pen[i] = uncertain.var(0)

                    ## attempt 1: the cos_dist. between the user and its neighbor in train set
                    elif uncertain_type == 'I-cos_min':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].min()
                    elif uncertain_type == 'I-cos_avg':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].mean()
                    elif uncertain_type == 'I-cos_max':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].max()

                    ## attempt 2: the cos_dist. between the user and its neighbor in block1and3 (indicator factor)
                    ## we weight the predicted r with cos_similarity
                    elif uncertain_type == "II-weight":
                        uncertain_weights = torch.tensor(1-cos_dist_ku[i]).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights * rewards, dim=0)
                    elif uncertain_type == "II-weight-div":
                        uncertain_weights = torch.tensor(1-cos_dist_ku[i]).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights * rewards, dim=0)/kr
                    elif uncertain_type == "II-weight-norm":
                        uncertain_weights_norm = torch.tensor((1-cos_dist_ku[i])/(1-cos_dist_ku[i]).sum()).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights_norm * rewards, dim=0)

                pred_cross_mat = cosine_similarity(pred_reward, pred_reward)
                topkr_indices = np.argsort(pred_cross_mat, axis=1)[:, -kr:]
                topku_indices = np.argsort(pred_cross_mat, axis=1)[:, -ku:]
                topku_sim = np.sort(pred_cross_mat, axis=1)[:, -ku:]
                cos_dist_ku = 1-topku_sim+1e-9

        return pred_reward, uncertain_pen

    elif env == 'CoatEnv-v0':
        with open('./saved_models/CoatEnv-v0/DeepFM/matsPre/[pointneg]_matPre.pickle', 'rb') as f:
            matpre = pickle.load(f)

        train_data_path = 'environments/coat/train.ascii'
        train_data = pd.read_csv(train_data_path, sep="\s+", header=None, dtype=str).to_numpy(dtype=int)

        ## GT for analysis only
        filename_GT = 'environments/RL4Rec/data/coat_pseudoGT_ratingM.ascii'
        mat = pd.read_csv(filename_GT, sep="\s+", header=None, dtype=str).to_numpy(dtype=int)

        pred_reward = np.zeros_like(matpre, dtype=float)
        uncertain_pen = np.zeros_like(matpre, dtype=float)

        if init_data == 'offline':
            pred_cross_mat = cosine_similarity(train_data, train_data)
            topkr_indices = np.argsort(pred_cross_mat, axis=1)[:, -kr:]
            topku_indices = np.argsort(pred_cross_mat, axis=1)[:, -ku:]

            topku_sim = np.sort(pred_cross_mat, axis=1)[:, -ku:]
            cos_dist_ku = 1-topku_sim+1e-9
            num_test_user, num_test_item = train_data.shape

            for j in range(num_iter):
                for i in range(len(topkr_indices)):
                    sim_users_kr = topkr_indices[i,:]
                    sim_users_ku = topku_indices[i,:]

                    rewards = train_data[sim_users_kr,:]
                    pred_reward[i] = rewards.mean(0)

                    ## Feb2024KDD Setting
                    if uncertain_type == 'I-var':
                        uncertain = train_data[sim_users_ku,:]
                        uncertain_pen[i] = uncertain.var(0)

                    ## attempt 1: the cos_dist. between the user and its neighbor in train set
                    elif uncertain_type == 'I-cos_min':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].min()
                    elif uncertain_type == 'I-cos_avg':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].mean()
                    elif uncertain_type == 'I-cos_max':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].max()

                    ## attempt 2: the cos_dist. between the user and its neighbor in block1and3 (indicator factor)
                    ## we weight the predicted r with cos_similarity
                    elif uncertain_type == "II-weight":
                        uncertain_weights = torch.tensor(1-cos_dist_ku[i]).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights * rewards, dim=0)
                    elif uncertain_type == "II-weight-div":
                        uncertain_weights = torch.tensor(1-cos_dist_ku[i]).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights * rewards, dim=0)/kr
                    elif uncertain_type == "II-weight-norm":
                        uncertain_weights_norm = torch.tensor((1-cos_dist_ku[i])/(1-cos_dist_ku[i]).sum()).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights_norm * rewards, dim=0)

                pred_cross_mat = cosine_similarity(pred_reward, pred_reward)
                topkr_indices = np.argsort(pred_cross_mat, axis=1)[:, -kr:]
                topku_indices = np.argsort(pred_cross_mat, axis=1)[:, -ku:]
                topku_sim = np.sort(pred_cross_mat, axis=1)[:, -ku:]
                cos_dist_ku = 1-topku_sim+1e-9

        elif init_data == 'wm':
            pred_cross_mat = cosine_similarity(matpre, matpre)
            topkr_indices = np.argsort(pred_cross_mat, axis=1)[:, -kr:]
            topku_indices = np.argsort(pred_cross_mat, axis=1)[:, -ku:]

            topku_sim = np.sort(pred_cross_mat, axis=1)[:, -ku:]
            cos_dist_ku = 1-topku_sim
            num_test_user, num_test_item = matpre.shape

            for j in range(num_iter):
                for i in range(len(topkr_indices)):
                    sim_users_kr = topkr_indices[i,:]
                    sim_users_ku = topku_indices[i,:]

                    rewards = matpre[sim_users_kr,:]
                    pred_reward[i] = rewards.mean(0)

                    ## Feb2024KDD Setting
                    if uncertain_type == 'I-var':
                        uncertain = matpre[sim_users_ku,:]
                        uncertain_pen[i] = uncertain.var(0)

                    ## attempt 1: the cos_dist. between the user and its neighbor in train set
                    elif uncertain_type == 'I-cos_min':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].min()
                    elif uncertain_type == 'I-cos_avg':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].mean()
                    elif uncertain_type == 'I-cos_max':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].max()

                    ## attempt 2: the cos_dist. between the user and its neighbor in block1and3 (indicator factor)
                    ## we weight the predicted r with cos_similarity
                    elif uncertain_type == "II-weight":
                        uncertain_weights = torch.tensor(1-cos_dist_ku[i]).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights * rewards, dim=0)
                    elif uncertain_type == "II-weight-div":
                        uncertain_weights = torch.tensor(1-cos_dist_ku[i]).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights * rewards, dim=0)/kr
                    elif uncertain_type == "II-weight-norm":
                        uncertain_weights_norm = torch.tensor((1-cos_dist_ku[i])/(1-cos_dist_ku[i]).sum()).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights_norm * rewards, dim=0)

                pred_cross_mat = cosine_similarity(pred_reward, pred_reward)
                topkr_indices = np.argsort(pred_cross_mat, axis=1)[:, -kr:]
                topku_indices = np.argsort(pred_cross_mat, axis=1)[:, -ku:]
                topku_sim = np.sort(pred_cross_mat, axis=1)[:, -ku:]
                cos_dist_ku = 1-topku_sim

        return pred_reward, uncertain_pen
    
    elif env == 'YahooEnv-v0':
        with open('./saved_models/YahooEnv-v0/DeepFM/matsPre/[pointneg]_matPre.pickle', 'rb') as f:
            matpre = pickle.load(f)

        ## GT for analysis only
        filename_GT = 'environments/RL4Rec/data/yahoo_pseudoGT_ratingM.ascii'
        mat = pd.read_csv(filename_GT, sep="\s+", header=None, dtype=str).to_numpy(dtype=int)
        mat = mat[:5400,:]

        train_data_path = 'environments/YahooR3/ydata-ymusic-rating-study-v1_0-train.txt'
        df_train, lbe_user, lbe_item, _ = get_df_yahoo(train_data_path)
        mat_train = csr_matrix((df_train['rating'], (df_train["user_id"], df_train["item_id"])),
                               shape=(df_train['user_id'].max() + 1, df_train['item_id'].max() + 1)).toarray()
        mat_train = mat_train[:5400,:]

        pred_reward = np.zeros_like(matpre, dtype=float)
        uncertain_pen = np.zeros_like(matpre, dtype=float)

        if init_data == 'offline':
            pred_cross_mat = cosine_similarity(mat_train, mat_train)
            topkr_indices = np.argsort(pred_cross_mat, axis=1)[:, -kr:]
            topku_indices = np.argsort(pred_cross_mat, axis=1)[:, -ku:]

            topku_sim = np.sort(pred_cross_mat, axis=1)[:, -ku:]
            cos_dist_ku = 1-topku_sim+1e-9
            num_test_user, num_test_item = mat_train.shape

            for j in range(num_iter):
                for i in range(len(topkr_indices)):
                    sim_users_kr = topkr_indices[i,:]
                    sim_users_ku = topku_indices[i,:]

                    rewards = mat_train[sim_users_kr,:]
                    pred_reward[i] = rewards.mean(0)

                    ## Feb2024KDD Setting
                    if uncertain_type == 'I-var':
                        uncertain = mat_train[sim_users_ku,:]
                        uncertain_pen[i] = uncertain.var(0)

                    ## attempt 1: the cos_dist. between the user and its neighbor in train set
                    elif uncertain_type == 'I-cos_min':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].min()
                    elif uncertain_type == 'I-cos_avg':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].mean()
                    elif uncertain_type == 'I-cos_max':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].max()

                    ## attempt 2: the cos_dist. between the user and its neighbor in block1and3 (indicator factor)
                    ## we weight the predicted r with cos_similarity
                    elif uncertain_type == "II-weight":
                        uncertain_weights = torch.tensor(1-cos_dist_ku[i]).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights * rewards, dim=0)
                    elif uncertain_type == "II-weight-div":
                        uncertain_weights = torch.tensor(1-cos_dist_ku[i]).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights * rewards, dim=0)/kr
                    elif uncertain_type == "II-weight-norm":
                        uncertain_weights_norm = torch.tensor((1-cos_dist_ku[i])/(1-cos_dist_ku[i]).sum()).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights_norm * rewards, dim=0)

                pred_cross_mat = cosine_similarity(pred_reward, pred_reward)
                topkr_indices = np.argsort(pred_cross_mat, axis=1)[:, -kr:]
                topku_indices = np.argsort(pred_cross_mat, axis=1)[:, -ku:]
                topku_sim = np.sort(pred_cross_mat, axis=1)[:, -ku:]
                cos_dist_ku = 1-topku_sim+1e-9

        elif init_data == 'wm':
            pred_cross_mat = cosine_similarity(matpre, matpre)
            topkr_indices = np.argsort(pred_cross_mat, axis=1)[:, -kr:]
            topku_indices = np.argsort(pred_cross_mat, axis=1)[:, -ku:]

            topku_sim = np.sort(pred_cross_mat, axis=1)[:, -ku:]
            cos_dist_ku = 1-topku_sim
            num_test_user, num_test_item = matpre.shape

            for j in range(num_iter):
                for i in range(len(topkr_indices)):
                    sim_users_kr = topkr_indices[i,:]
                    sim_users_ku = topku_indices[i,:]

                    rewards = matpre[sim_users_kr,:]
                    pred_reward[i] = rewards.mean(0)

                    ## Feb2024KDD Setting
                    if uncertain_type == 'I-var':
                        uncertain = matpre[sim_users_ku,:]
                        uncertain_pen[i] = uncertain.var(0)

                    ## attempt 1: the cos_dist. between the user and its neighbor in train set
                    elif uncertain_type == 'I-cos_min':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].min()
                    elif uncertain_type == 'I-cos_avg':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].mean()
                    elif uncertain_type == 'I-cos_max':
                        uncertain_pen[i] = torch.ones(num_test_item)*cos_dist_ku[i].max()

                    ## attempt 2: the cos_dist. between the user and its neighbor in block1and3 (indicator factor)
                    ## we weight the predicted r with cos_similarity
                    elif uncertain_type == "II-weight":
                        uncertain_weights = torch.tensor(1-cos_dist_ku[i]).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights * rewards, dim=0)
                    elif uncertain_type == "II-weight-div":
                        uncertain_weights = torch.tensor(1-cos_dist_ku[i]).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights * rewards, dim=0)/kr
                    elif uncertain_type == "II-weight-norm":
                        uncertain_weights_norm = torch.tensor((1-cos_dist_ku[i])/(1-cos_dist_ku[i]).sum()).unsqueeze(1)
                        pred_reward[i] = torch.sum(uncertain_weights_norm * rewards, dim=0)

                pred_cross_mat = cosine_similarity(pred_reward, pred_reward)
                topkr_indices = np.argsort(pred_cross_mat, axis=1)[:, -kr:]
                topku_indices = np.argsort(pred_cross_mat, axis=1)[:, -ku:]
                topku_sim = np.sort(pred_cross_mat, axis=1)[:, -ku:]
                cos_dist_ku = 1-topku_sim

        return pred_reward, uncertain_pen
    




# print(1)
if __name__ == '__main__':
    # rand_r_mat, rand_var_mat = knn_based_pred_reward("KuaiRand-v0", 150, 150, 'II-weight-norm', 'wm', 5)
    # rec_r_mat, rec_var_mat = knn_based_pred_reward("KuaiEnv-v0", 50, 50, 'II-weight-norm')
    # coat_r_mat, coat_var_mat = knn_based_pred_reward("CoatEnv-v0", 15, 15, 'II-weight-norm', 'wm', 10)
    yahoo_r_mat, yahoo_var_mat = knn_based_pred_reward("YahooEnv-v0", 20, 20, 'II-weight-norm', 'wm', 5)