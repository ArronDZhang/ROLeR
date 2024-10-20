import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from vis import svd_reduction_and_visualized_with_color, just_svd

## Don YZ: the up-to-date visualization of user embedding

def vis_group(full_embs, group, color, label):
    group_embs = full_embs[group.index]
    plt.scatter(group_embs[:, 0], group_embs[:, 1], color=color, label=label, marker='.')

def plot_all_groups(reduced_user_embs, groups, colors, labels):
    plt.figure()

    for group, color, label in zip(groups, colors, labels):
        vis_group(reduced_user_embs, group, color, label)

    # Balance the axes
    all_embs = np.concatenate([reduced_user_embs[group.index] for group in groups])
    max_range = max(np.abs(all_embs).max(), np.abs(all_embs).max())
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)

    # Center the origin and add grid
    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)
    plt.grid(True)

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('2D Visualization of the Reduced Data')

    # Add legend
    plt.legend()

    plt.show()
    plt.close()

if __name__ == '__main__':
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

    # Define colors for each range
    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#00ffff"]

    # Group the data
    groups = [group1k, group2k, group3k, group4k, group5k]

    labels = ["0-1000", "1000-2000", "2000-3000", "3000-4000", "4000-5000"]

    # Plot all groups with specified colors
    plot_all_groups(reduced_user_embs, groups, colors, labels)
