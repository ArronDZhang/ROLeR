import torch
import os
import pandas as pd
from vis import svd_reduction_and_visualized_with_color, just_svd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

WINDOW_SIZE = 3
big_matrix_processed, small_matrix_processed = 'environments/KuaiRec/data/big_matrix_processed.csv', \
    'environments/KuaiRec/data/small_matrix_processed.csv'

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

rew_big, rew_small = load_mat(big_matrix_processed), load_mat(small_matrix_processed)
saved_model = torch.load("/home/s4715423/DORL-codes/saved_models/KuaiEnv-v0/A2C_with_emb/A2C_with_emb_DORL-e200.pt")

# def select_seq(source, target, window_size=WINDOW_SIZE):
#     selected = []
#     for i in range(len(source)-3):
#         if source[i] in target and source[i+1] in target and source[i+2] in target and source[i+3] in target:
#             selected.append((source[i], source[i+1], source[i+2], source[i+3]))
#     return selected

big_matrix, small_matrix = pd.read_csv('environments/KuaiRec/data/big_matrix_processed.csv'), \
    pd.read_csv('environments/KuaiRec/data/small_matrix_processed.csv')

big_matrix_dict, small_matrix_dict = big_matrix.groupby('user_id')['item_id'].apply(list).to_dict(), \
    small_matrix.groupby('user_id')['item_id'].apply(list).to_dict()

# target_lst = small_matrix['item_id'].to_list()

state_tracker = saved_model["state_tracker"]
feat_user, feat_item = state_tracker.values()
feat_item = feat_item[1:,:]

feat_item = just_svd(feat_item.cpu())

big_matrix_dict_embs = {key: [] for key in big_matrix_dict.keys()}

for key, value in big_matrix_dict.items():
    for i in range(len(value)):
        if value[i] < 3327:
            big_matrix_dict_embs[key].append(feat_item[value[i]])

big_matrix_dict_embs1 = {key:value for key, value in big_matrix_dict_embs.items() if key<1411} # {user:[embs]}

small_matrix_dict_embs = {key: [] for key in small_matrix_dict.keys()}
for key, value in small_matrix_dict.items():
    for i in range(len(value)):
        if value[i] < 3327:
            small_matrix_dict_embs[key].append(feat_item[value[i]])

small_matrix_dict_embs1 = {key:value for key, value in small_matrix_dict_embs.items() if key<1411}

big_matrix_dict1, small_matrix_dict1 = {key:[] for key in big_matrix_dict.keys()}, \
    {key:[] for key in small_matrix_dict.keys()} 
for key, value in big_matrix_dict.items():
    for i in range(len(value)):
        if value[i] < 3327:
            big_matrix_dict1[key].append(value[i])

for key, value in small_matrix_dict.items():
    for i in range(len(value)):
        if value[i] < 3327:
            small_matrix_dict1[key].append(value[i])

big_matrix_dict1, small_matrix_dict1  = {key:value for key, value in big_matrix_dict1.items() if key<1411}, \
    {key:value for key, value in small_matrix_dict1.items() if key<1411} # {user:[items]}


# def vis(item_dic, emb_dic, rew_mat):
#     plt.figure()
#     for key in emb_dic:
#         # Ensure the key exists in item_dic and rew_mat
#         if key in item_dic and key in rew_mat:
#             positions = emb_dic[key]
#             items = item_dic[key]

#             # Process in chunks of 3
#             for i in range(0, len(positions), 3):
#                 # Get the chunk of three elements
#                 pos_chunk = positions[i:i+3]
#                 item_chunk = items[i:i+3]

#                 # Check if the chunk is of size 3, otherwise break
#                 if len(pos_chunk) < 3:
#                     break

#                 # Calculate averaged position
#                 avg_pos = np.mean(pos_chunk, axis=0)

#                 # Calculate total reward for the chunk
#                 reward = sum(rew_mat[key][item] for item in item_chunk)

#                 # Plot the point with color intensity based on the reward
#                 plt.scatter(avg_pos[0], avg_pos[1], c='red', alpha=reward/10, edgecolors='none')

#     plt.colorbar(label='Reward Intensity')
#     plt.xlabel('Position X')
#     plt.ylabel('Position Y')
#     plt.title('Averaged Position and Reward Intensity Plot')
#     plt.show()

# Example usage:
# Assuming item_dic, emb_dic, and rew_mat are defined as per your description
# vis(item_dic, emb_dic, rew_mat)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def vis(item_dic, emb_dic, rew_mat):
    # Create a figure and an axes
    fig, ax = plt.subplots()

    # Find the maximum reward for normalization
    max_reward = max(sum(rew_mat[key][item] for item in items) for key, items in item_dic.items())

    # Use the 'magma_r' colormap
    # cmap = plt.cm.magma_r
    cmap = plt.cm.magma

    for key in emb_dic:
        if key in item_dic and key in rew_mat:
            positions = emb_dic[key]
            items = item_dic[key]

            for i in range(0, len(positions), 3):
                pos_chunk = positions[i:i+3]
                item_chunk = items[i:i+3]

                if len(pos_chunk) < 3:
                    break

                avg_pos = np.mean(pos_chunk, axis=0)
                reward = sum(rew_mat[key][item] for item in item_chunk)

                # Normalize the reward to be between 0 and 1
                normalized_reward = reward / max_reward

                # Use the normalized reward to determine the color
                ax.scatter(avg_pos[0], avg_pos[1], color=cmap(normalized_reward), edgecolors='none')

    # Create a color bar
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Reward Intensity')

    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    # ax.set_title('Averaged Position and Reward Intensity Plot')
    plt.show()
    plt.savefig('train_set.png')
    plt.close()

# Example usage:
# vis(item_dic, emb_dic, rew_mat)



# vis(big_matrix_dict1, big_matrix_dict_embs1, rew_big)
# vis(big_matrix_dict1, big_matrix_dict_embs1, rew_small)
vis(small_matrix_dict1, small_matrix_dict_embs1, rew_small)
# vis(small_matrix_dict1, small_matrix_dict_embs1, rew_small)