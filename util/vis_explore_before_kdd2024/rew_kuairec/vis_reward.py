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

## Don YZ: the up-to-date comprehensive visualization about the GT reward and DeepFM estimated reward on KuaiRec

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

## prepare trajectory, ground truth reward and policy estimated reward
file = 'environments/KuaiRec/data/small_matrix_processed.csv'
trajectory_matrix = pd.read_csv(file, header=0)
rew_gt, lbe_user, lbe_item = load_mat(file)
rew_diff = rew_gt-matpre

# plt.imshow(np.abs(rew_gt-matpre), cmap='coolwarm', interpolation='nearest')
# plt.colorbar(label='Difference')
# plt.title('Element-wise Difference Between Two Matrices')
# plt.xlabel('Column')
# plt.ylabel('Row')
# plt.show()

# Using Seaborn heatmap
sns.heatmap(rew_diff, annot=False, cmap='coolwarm', center=0)
plt.title('Element-wise Difference Between Two Matrices')
plt.show()
plt.close()

## here we analysis the reward diff
## 1) reward diff hist 
bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
counts, edges = np.histogram(rew_diff, bins)
widths = np.diff(edges)
plt.bar(edges[:-1], counts/counts.sum(), width=widths, align='edge', edgecolor='black', alpha=0.7)
plt.title('Proportion of Each Group')
plt.xlabel('Range')
plt.ylabel('Proportion')
plt.show()
plt.close()

## 2) user-side and item-side reward diff hist
user_bins = rew_diff.mean(1)
user_max, user_min = user_bins.max(), user_bins.min()
user_bin = [0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
count_user, _ = np.histogram(user_bins, user_bin)
widths_user = np.diff(user_bin)
plt.bar(user_bin[:-1], count_user/count_user.sum(), width=widths_user, align='edge', edgecolor='black', alpha=0.7)
plt.title('User Proportion of Avg. Reward Diff x All Items')
plt.xlabel('Range')
plt.ylabel('Proportion')
plt.show()
plt.close()

item_bins = rew_diff.mean(0)
item_max, item_min = item_bins.max(), item_bins.min()
item_bin = [0, 0.5, 1, 1.5, 2, 2.5, 3]
count_item, _ = np.histogram(item_bins, item_bin)
widths_item = np.diff(item_bin)
plt.bar(item_bin[:-1], count_item/count_item.sum(), width=widths_item, align='edge', edgecolor='black', alpha=0.7)
plt.title('Item Proportion of Avg. Reward Diff x All Users')
plt.xlabel('Range')
plt.ylabel('Proportion')
plt.show()
plt.close()

## 3) statistic of small matrix related users and items
user_features, item_features, reward_features = get_features("KuaiEnv-v0")
## obtain clean data
df_train, df_user, df_item, list_feat = get_training_data("KuaiEnv-v0") 
# lbe_user, lbe_item = KuaiEnv.get_lbe() # these two LabelEncoders can be obtained in load_mat(), the same

## obtain testing user and testing item related df_train
user_small_ary, item_small_ary = lbe_user.classes_, lbe_item.classes_
df_user_small = df_train[df_train['user_id'].isin(user_small_ary)]
df_item_small = df_train[df_train['item_id'].isin(item_small_ary)]

## obtain user activeness and item populrity
user_activeness = df_user_small[['user_id', 'item_id']].groupby('user_id').count().reset_index()
user_activeness = user_activeness.rename(columns={"item_id":"activeness"})
user_activeness['user_id'] = lbe_user.transform(user_activeness['user_id'])
activeness_max, activeness_min = user_activeness['activeness'].max(), user_activeness['activeness'].min()
## 154~1186
activeness_bin = [i for i in range(100, 1300, 100)]
## 3.1) user activeness distribution
activeness_dist, _ = np.histogram(user_activeness['activeness'], activeness_bin)
widths_activeness = np.diff(activeness_bin)
plt.bar(activeness_bin[:-1], activeness_dist/activeness_dist.sum(), width=widths_activeness, align='edge', edgecolor='black', alpha=0.7)
plt.title('User Activeness Distribution')
plt.xlabel('Activeness')
plt.ylabel('Proportion')
plt.show()
plt.close()

## 3.1_plus) user activeness distribution
sns.kdeplot(user_activeness['activeness'], fill=True)
plt.title('User Activeness')
plt.ylabel('Density')
plt.show()
plt.close()

activeness_bin_new = [i for i in range(150, 1200, 10)]
activeness_dist_new, _ = np.histogram(user_activeness['activeness'], activeness_bin_new)
widths_activeness_new = np.diff(activeness_bin_new)
plt.bar(activeness_bin_new[:-1], activeness_dist_new/activeness_dist_new.sum(), \
    width=widths_activeness_new, align='edge', edgecolor='black', alpha=0.7)
plt.title('User Activeness Distribution')
plt.xlabel('Activeness')
plt.ylabel('Proportion')
plt.show()
plt.close()

item_popularity = df_item_small[['user_id', 'item_id']].groupby('item_id').count().reset_index()
item_popularity = item_popularity.rename(columns={"user_id":"popularity"})
item_popularity['item_id'] = lbe_item.transform(item_popularity['item_id'])
popularity_max, popularity_min = item_popularity['popularity'].max(), item_popularity['popularity'].min()
## 4~27615
popularity_bin = [i for i in range(0, 7000, 500)]
## 3.2) item popularity distribution
popularity_dist, _ = np.histogram(item_popularity['popularity'], popularity_bin)
widths_popularity = np.diff(popularity_bin)
plt.bar(popularity_bin[:-1], popularity_dist/popularity_dist.sum(), width=widths_popularity, align='edge', edgecolor='black', alpha=0.7)
plt.title('Item Popularity Distribution')
plt.xlabel('Popularity')
plt.ylabel('Proportion')
plt.show()
plt.close()

## 3.3) item tag distribution
## note that the tages in list_feat is inconsistant with that in df_item: df_item-1=list_feat
# item_tag_small = df_item.loc[item_small_ary][['feat0', 'feat1', 'feat2', 'feat3']]
# item_tag_dic = item_tag_small.T.to_dict('list')
tag_small_lst = [list_feat[i] for i in item_small_ary]
flatten_tag_small = [tag for lst in tag_small_lst for tag in lst]
tag_stat = Counter(flatten_tag_small)
key, value = tag_stat.keys(), tag_stat.values()
plt.bar(key, value)
plt.title('Tag Frequencies')
plt.xlabel('Tag')
plt.ylabel('Frequency')
plt.show()
plt.close()

## 4) combine part3 with reward diff
## 4.1) activeness X rew_diff
user_activeness['mean_rew_diff'] = user_bins
activeness_level_rew_diff = {}
for i in range(len(activeness_bin)-1):
    level_i_user = user_activeness[(user_activeness['activeness'] > activeness_bin[i]) & \
        (user_activeness['activeness'] > activeness_bin[i])]
    activeness_level_rew_diff[i+1] = level_i_user['mean_rew_diff'].mean()

## 4.2) popularity X rew_diff
item_popularity['mean_rew_diff'] = item_bins
popularity_level_rew_diff = {}
for i in range(len(popularity_bin)-1):
    level_i_item = item_popularity[(item_popularity['popularity'] > popularity_bin[i]) & \
        (item_popularity['popularity'] > popularity_bin[i])]
    popularity_level_rew_diff[i+1] = level_i_item['mean_rew_diff'].mean()

## 4.3) tag X rew_diff
item_tag_small = df_item.loc[item_small_ary][['feat0', 'feat1', 'feat2', 'feat3']].reset_index()
item_tag_small['item_id'] = lbe_item.transform(item_tag_small['item_id'])
melted_item_tag = item_tag_small.melt(id_vars='item_id', value_vars=['feat0', 'feat1', 'feat2', 'feat3'], \
    var_name='feature', value_name='value')

# filter out rows where the feature value is 0
melted_item_tag = melted_item_tag[melted_item_tag['value'] != 0]
item_tag = melted_item_tag.groupby('value')['item_id'].apply(list).to_dict()

# be consistent with list_feat
tag_item = {}
for key, value in item_tag.items():
    rew_diff_lst = []
    for i in range(len(value)):
        rew_diff_lst.append(float(item_popularity[item_popularity['item_id']==value[i]]['mean_rew_diff']))
    tag_item[key-1] = np.mean(rew_diff_lst)

## 4.4) vis 4.1~4.3
def dict_to_plot(dic, title, xlabel, ylabel):
    key, value = dic.keys(), dic.values()
    plt.bar(key, value)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.close()

dict_to_plot(activeness_level_rew_diff, 'User Activeness Level Reward Diff.', 'User Activeness Level', 'Reward Diff.')
dict_to_plot(popularity_level_rew_diff, 'Item Popularity Level Reward Diff.', 'Item Popularity Level', 'Reward Diff.')
dict_to_plot(tag_item, 'Reward Diff. for each Tag', 'Tags', 'Reward Diff.')

## 4.1_plus) activeness X rew_diff
# keys = list(user_activeness['activeness'])
# values = list(user_activeness['mean_rew_diff'])
# plt.scatter(keys, values)  # 'marker' adds a dot at each data point
# plt.title('User Activeness Level Reward Diff.')
# plt.xlabel('Activeness')
# plt.ylabel('Rew_Diff.')
# plt.show()
# plt.close()

user_activeness['mean_rew_diff'] = user_bins
activeness_level_rew_diff_new = {}
for i in range(len(activeness_bin_new)-1):
    level_i_user = user_activeness[(user_activeness['activeness'] > activeness_bin_new[i]) & \
        (user_activeness['activeness'] > activeness_bin_new[i])]
    activeness_level_rew_diff_new[i+1] = level_i_user['mean_rew_diff'].mean()

dict_to_plot(activeness_level_rew_diff_new, 'User Activeness Level Reward Diff.', 'User Activeness Level', 'Reward Diff.')

print(1)

