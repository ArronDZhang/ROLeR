import torch
import os
import pandas as pd
from vis import svd_reduction_and_visualized_with_color, just_svd
import matplotlib.pyplot as plt
import numpy as np

IS_FILTER = True

user_embedding = torch.load('/home/s4715423/DORL-codes/saved_models/KuaiEnv-v0/DeepFM/embeddings/[pointneg]_emb_user_M0.pt')
embedding_layer = torch.nn.Embedding.from_pretrained(user_embedding, freeze=True)

rootpath= os.path.join("/home/s4715423/DORL-codes", "environments", "KuaiRec")
# small_matrix = pd.read_csv(rootpath + "/data/small_matrix.csv")
big_matrix = pd.read_csv(rootpath + "/data/big_matrix.csv")

user_count_train = big_matrix[["user_id", "video_id"]].groupby("user_id").count()# .reset_index()，这里reset后续操作index会有问题
user_count_train = user_count_train.rename(columns={"video_id":"vitality"})

user_ids = torch.tensor(user_count_train.index)
user_embs = embedding_layer(user_ids)

reduced_user_embs = just_svd(user_embs)

## grouping
group1k, group2k, group3k, group4k, group5k = user_count_train[user_count_train['vitality']<1000], \
    user_count_train[(user_count_train['vitality']>=1000) & (user_count_train['vitality']<2000)], \
        user_count_train[(user_count_train['vitality']>=2000) & (user_count_train['vitality']<3000)], \
            user_count_train[(user_count_train['vitality']>=3000) & (user_count_train['vitality']<4000)], \
                user_count_train[(user_count_train['vitality']>=4000) & (user_count_train['vitality']<5000)]

def vis_group(full_embs, group, color_column):
    group_embs = full_embs[group.index]
    plt.scatter(group_embs[:, 0], group_embs[:, 1], c=group[color_column], cmap='magma_r', marker='.')
    plt.colorbar(label=color_column)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Balance the axes
    max_range = max(np.abs(group_embs).max(), np.abs(group_embs).max())
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)

    # Center the origin
    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)
    plt.grid(True)

    plt.title('2D Visualization of the Reduced Data with Color Coding')
    plt.show()
    plt.close()

if __name__ == '__main__':
    vis_group(reduced_user_embs, group1k, 'vitality')
    vis_group(reduced_user_embs, group2k, 'vitality')
    vis_group(reduced_user_embs, group3k, 'vitality')
    vis_group(reduced_user_embs, group4k, 'vitality')
    vis_group(reduced_user_embs, group5k, 'vitality')