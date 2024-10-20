import torch
import os
import pandas as pd
from vis import svd_reduction_and_visualized_with_color

saved_model = torch.load("/home/s4715423/DORL-codes/saved_models/KuaiEnv-v0/A2C_with_emb/A2C_with_emb_DORL-e200.pt")
state_tracker = saved_model["state_tracker"]
feat_user, feat_item = state_tracker.values()
feat_item = feat_item[1:,:]

rootpath= os.path.join("/home/s4715423/DORL-codes", "environments", "KuaiRec")
small_matrix = pd.read_csv(rootpath + "/data/small_matrix.csv")
big_matrix = pd.read_csv(rootpath + "/data/big_matrix.csv")

item_count_train = big_matrix[["user_id", "video_id"]].groupby("video_id").count().reset_index()
item_count_train = item_count_train.rename(columns={"user_id":"count"})
item_count_test = small_matrix[["user_id", "video_id"]].groupby("video_id").count().reset_index()

item_train_subset = item_count_train.merge(item_count_test, on="video_id", how='inner')

svd_reduction_and_visualized_with_color(feat_item.cpu(), item_train_subset)
