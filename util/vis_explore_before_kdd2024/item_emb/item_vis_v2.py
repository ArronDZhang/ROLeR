import torch
import os
import pandas as pd
from vis import svd_reduction_and_visualized_with_color, just_svd
import matplotlib.pyplot as plt
import numpy as np

IS_FILTER = True

user_embedding = torch.load('/home/s4715423/DORL-codes/saved_models/KuaiEnv-v0/DeepFM/embeddings/[pointneg]_emb_item_M0.pt')
embedding_layer = torch.nn.Embedding.from_pretrained(user_embedding, freeze=True)

rootpath= os.path.join("/home/s4715423/DORL-codes", "environments", "KuaiRec")
# small_matrix = pd.read_csv(rootpath + "/data/small_matrix.csv")
big_matrix = pd.read_csv(rootpath + "/data/big_matrix.csv")

item_count_train = big_matrix[["user_id", "video_id"]].groupby("video_id").count()# .reset_index()，这里reset后续操作index会有问题
item_count_train = item_count_train.rename(columns={"user_id":"count"})

## filter
item_count_train = item_count_train[item_count_train['count']<=4000]

item_ids = torch.tensor(item_count_train.index)
item_embs = embedding_layer(item_ids)

svd_reduction_and_visualized_with_color(item_embs.cpu(), item_count_train, 'count')
