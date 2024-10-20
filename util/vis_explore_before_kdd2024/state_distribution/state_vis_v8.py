import torch
import os
import pandas as pd
from vis import svd_reduction_and_visualized_with_color, just_svd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

## It's the standard version for visualization of state distribution 

WINDOW_SIZE = 50

## trained embedding then svd
saved_model = torch.load("/home/s4715423/DORL-codes/saved_models/KuaiEnv-v0/A2C_with_emb/A2C_with_emb_DORL-e200.pt")
state_tracker = saved_model["state_tracker"]
feat_user, feat_item = state_tracker.values()
feat_item = feat_item[:-1,:]
reduced_item = just_svd(feat_item.cpu())

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

    return mat

## prepare trajectory, ground truth reward and policy estimated reward
file = 'environments/KuaiRec/data/small_matrix_processed.csv'
trajectory_matrix = pd.read_csv(file, header=0)
rew_gt = load_mat(file)
trajectory_dict = trajectory_matrix.groupby('user_id')['item_id'].apply(list).to_dict()

## handle user and item index map
item_set = set(trajectory_matrix['item_id']) # if write in one line, time consuming
item_map = {sorted(list(item_set))[i]:i for i in range(len(item_set))}
user_set = set(trajectory_matrix['user_id'])
user_map = {sorted(list(user_set))[i]:i for i in range(len(user_set))}


def split_tra_map_item(lst, item_map=item_map, window_size=WINDOW_SIZE):
    new_lst = []
    for i in range(0, len(lst), window_size):
        new_lst.append(list(map(item_map.get, lst[i:i+window_size])))
    return new_lst

class Point:
    def __init__(self, user_id, item_ids, item_embs, rewards):
        self.user_id = user_id
        self.item_ids = item_ids
        self.item_embs = item_embs
        self.rewards = rewards
    
    def aggr_embs(self):
        return np.mean(self.item_embs, axis=0)
    
    def aggr_rew(self, way):
        if way == 'avg':
            return np.mean(self.rewards)
        elif way == 'sum':
            return np.sum(self.rewards)
        else:
            NotImplemented

## store all data as Points, seems too slow in this way
points = []
for key, value in trajectory_dict.items():
    items_lst = split_tra_map_item(trajectory_dict[key])
    for i in range(len(items_lst)):
        rewards = rew_gt[user_map[key], items_lst[i]]
        positions = reduced_item[items_lst[i]]
        points.append(Point(user_map[key], items_lst[i], positions, rewards))

def vis(points):
    positions = np.array(list(point.aggr_embs() for point in points))
    rews = np.array(list(point.aggr_rew('avg') for point in points))
    plt.scatter(positions[:,0], positions[:,1], c=rews, cmap='magma_r', marker='.')
    plt.colorbar(label=rews)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Balance the axes
    max_range = max(np.abs(reduced_item).max(), np.abs(reduced_item).max())
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)

    # Center the origin
    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)
    plt.grid(True)


    plt.title('State Distribution')
    plt.savefig('State Distribution.png')
    plt.show()
    plt.close()

if __name__ == '__main__':
    vis(points)