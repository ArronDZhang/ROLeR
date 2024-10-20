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

def get_args_all():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument("--read_message", type=str, default="pointneg")
    # parser.add_argument("--message", type=str, default="A2C_with_emb")
    args = parser.parse_known_args()[0]
    return args

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


def prepare_user_model_and_env(args):
    args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    np.random.seed(2022)
    random.seed(2022)

    UM_SAVE_PATH = os.path.join(".", "saved_models", "KuaiEnv-v0", "DeepFM")
    # MODEL_MAT_PATH = os.path.join(UM_SAVE_PATH, "mats", f"[{args.read_message}]_mat.pickle")
    MODEL_PARAMS_PATH = os.path.join(UM_SAVE_PATH, "params", f"[{args.read_message}]_params.pickle")

    with open(MODEL_PARAMS_PATH, "rb") as file:
        model_params = pickle.load(file)

    n_models = model_params["n_models"]
    model_params.pop('n_models')

    ensemble_models = EnsembleModel(n_models, args.read_message, UM_SAVE_PATH, **model_params)
    ensemble_models.load_all_models()

    return ensemble_models


## train and test matrices
file_train = 'environments/KuaiRec/data/big_matrix_processed.csv'
file_test = "environments/KuaiRec/data/small_matrix_processed.csv"

train_matrix, _, _ = load_mat(file_train)
test_matrix, lbe_user, lbe_item = load_mat(file_test)

args = get_args_all()
ensemble_models = prepare_user_model_and_env(args)
saved_embedding = ensemble_models.load_val_user_item_embedding(freeze_emb=True)
user_embedding = saved_embedding['feat_user']
# embedding_layer = torch.nn.Embedding.from_pretrained(user_embedding, freeze=True)
## 这里只load了small matrix相关的embedding
## 去看ensemble_model中load embedding的函数，实际上也只是load了第一个user model的embedding
## 于是新开一个文件，此版为仅供参考的废案
print(1)