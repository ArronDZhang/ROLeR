# -*- coding: utf-8 -*-
# @Time    : 2022/11/2 14:20
# @Author  : Chongming GAO
# @FileName: visual_kuairand_pure.py

import os
import pandas as pd
import numpy as np
import collections
import itertools
import ipdb

# from data_analysis.visual import visual_continue, visual_statistics_discrete, visual_with_hue
from visual import visual_continue, visual_statistics_discrete, visual_with_hue

rootpath= os.path.join("..", "environments", "KuaiRec")

print("Loading big matrix...")
big_matrix = pd.read_csv(rootpath + "/data/big_matrix.csv")
print("Loading small matrix...")
small_matrix = pd.read_csv(rootpath + "/data/small_matrix.csv")

print("Loading social network...")
social_network = pd.read_csv(rootpath + "/data/social_network.csv")
social_network["friend_list"] = social_network["friend_list"].map(eval)

print("Loading item features...")
item_categories = pd.read_csv(rootpath + "/data/item_categories.csv")
item_categories["feat"] = item_categories["feat"].map(eval)

print("Loading user features...")
user_features = pd.read_csv(rootpath + "/data/user_features.csv")

print("Loading items' daily features...")
item_daily_features = pd.read_csv(rootpath + "/data/item_daily_features.csv")

print("All data loaded.")

# num_feat = video_features_basic["tag"].map(len) # 数每一行有几个tag
# print(num_feat.describe())
# visual_statistics_discrete(num_feat, "视频标签数目")

df_train, df_test = big_matrix, small_matrix

df_train["domain"] = "Standard"
df_test["domain"] = "Random"
df_all = df_train._append(df_test)
df_all = df_all.reset_index(drop=True)


def get_pop(df_train, popbin):
    df_popularity = df_train[["item_id", "user_id"]].groupby("item_id").agg(len) # count出每个item有多少用户点击
    miss_id = list(set(range(df_train["item_id"].max() + 1)) - set(df_popularity.index)) # 找出没有用户点击的items
    df_miss = pd.DataFrame({"id": miss_id, "user_id": 0})
    df_miss.set_index("id", inplace=True)
    df_pop = df_popularity._append(df_miss)
    df_pop = df_pop.sort_index() # 不需要reset_index，因为两个df没有重复的index
    df_pop.rename(columns={"user_id": "count"}, inplace=True)

    # # for feat in df_train.columns[3:]:
    # #     df_feat_pop = df_train[[feat, "user_id"]].groupby(feat).agg(len)
    # #     print(df_feat_pop)
    #
    # feat = "age" # todo: for coat
    # df_feat_pop = df_train[[feat, "user_id"]].groupby(feat).agg(len)
    # print(df_feat_pop)

    # df_pop = df_pop.sort_values(by="count").reset_index(drop=True)

    bins = popbin + [df_pop.max()[0] + 1]
    pop_res = {} # 分桶并数出每个区间的数目
    for left, right in zip(bins[:-1], bins[1:]):
        df_pop.loc[df_pop["count"].map(lambda x: x >= left and x < right), "pop_group"] = "({},{})".format(left, right)
        # df_pop.loc[df_pop["count"].map(lambda x: x >= left and x < right), "pop_group"] = left
        pop_res["[{},{})".format(left,right)] = sum(df_pop["count"].map(lambda x: x >= left and x < right))

    print(pop_res)

    # sns.histplot(data=df_pop, x="count", bins=100)
    # plt.savefig(os.path.join(CODEPATH, f'dist_pop_{envname}.pdf'), bbox_inches='tight', pad_inches=0)
    # plt.close()

    return df_pop # 以item id为index的df，只有对应item的出现次数和所属分桶两列


df_train.rename(columns={"video_id":"item_id"}, inplace=True)
df_test.rename(columns={"video_id":"item_id"}, inplace=True)
df_all.rename(columns={"video_id":"item_id"}, inplace=True)

popbin = [0, 10, 20, 40, 80, 150, 300]
df_pop = get_pop(df_train, popbin)

df_pop["count"].describe()


# df_train["item_pop"] = df_pop["count"].loc[df_train["item_id"]].reset_index(drop=True)
# df_train["pop_group"] = df_pop["pop_group"].loc[df_train["item_id"]].reset_index(drop=True)
#
# df_rand["item_pop"] = df_pop["count"].loc[df_rand["item_id"]].reset_index(drop=True)
# df_rand["pop_group"] = df_pop["pop_group"].loc[df_rand["item_id"]].reset_index(drop=True)

visual_continue(df_pop["count"], var="商品流行度")

df_all["item_pop"] = df_pop["count"].loc[df_all["item_id"]].reset_index(drop=True)
# 这里的用法有点意思，loc不一定只用于筛选，也可用于扩充，类似outer join
df_all["pop_group"] = df_pop["pop_group"].loc[df_all["item_id"]].reset_index(drop=True)
# df_pop1 = df_pop.reset_index()
# df_all1 = df_all.merge(df_pop1, 'left', on='item_id')
# ipdb.set_trace()

# visual_continue(df_train["item_pop"], var="训练集流行度")
# visual_continue(df_rand["item_pop"], var="测试集流行度")
#
# visual_continue(df_train["pop_group"], var="训练集流行度分组", is_sort=True, xrotation=15)
# visual_continue(df_rand["pop_group"], var="测试集流行度分组", is_sort=True, xrotation=15)

visual_with_hue(df_all, var="流行度分组", x="pop_group", hue="domain", bin=100, is_sort=True, xrotation=15)
visual_with_hue(df_all, var="流行度", x="item_pop", hue="domain", bin=100, is_sort=False, xrotation=0)




# visual_continue(df_train["pop_group"], var="商品流行度")






