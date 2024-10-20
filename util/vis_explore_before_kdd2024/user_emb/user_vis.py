import torch
import os
import pandas as pd
from vis import svd_reduction_and_visualized_with_color
import numpy as np

IS_FILTER = True

user_embedding = torch.load('/home/s4715423/DORL-codes/saved_models/KuaiEnv-v0/DeepFM/embeddings/[pointneg]_emb_user_M0.pt')
embedding_layer = torch.nn.Embedding.from_pretrained(user_embedding, freeze=True)

rootpath= os.path.join("/home/s4715423/DORL-codes", "environments", "KuaiRec")
# small_matrix = pd.read_csv(rootpath + "/data/small_matrix.csv")
big_matrix = pd.read_csv(rootpath + "/data/big_matrix.csv")

user_count_train = big_matrix[["user_id", "video_id"]].groupby("user_id").count()# .reset_index()，这里reset后续操作index会有问题
user_count_train = user_count_train.rename(columns={"video_id":"vitality"})

user_count_train1 = big_matrix[["user_id", "video_id"]].groupby("user_id").count().reset_index()
user_count_train1 = user_count_train1.rename(columns={"video_id":"vitality"})

# 数了一下vitality>6000的一共5个，去掉它们
if IS_FILTER:
    user_count_train = user_count_train[user_count_train['vitality']<6000]
    user_count_train1 = user_count_train1[user_count_train1['vitality']<6000]


# user_ids = torch.tensor(user_count_train['user_id']) # 这时候index会不一致
## 上述问题本质原因是index不连续，直接转化成tensor会报错，一个方法是先转化成np.array过渡
user_ids = torch.tensor(user_count_train.index)
user_embs = embedding_layer(user_ids)

svd_reduction_and_visualized_with_color(user_embs.cpu(), user_count_train, 'vitality')

set0, set1 = list(user_count_train.index), list(user_count_train1["user_id"])
# print("=============================================")
# print("set difference:", np.mean(set0)-np.mean(set1))
# print("=============================================")