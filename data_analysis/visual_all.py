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

rootpath= os.path.join("..", "environments", "KuaiRand_Pure")

df_rand = pd.read_csv(os.path.join(rootpath, "data/log_random_4_22_to_5_08_pure.csv"), nrows=10000000)
# ipdb.set_trace()
# 这个数据集是纯random rec得到的，其每行数据的is_rand均为1

df1 = pd.read_csv(os.path.join(rootpath, "data/log_standard_4_08_to_4_21_pure.csv"), nrows=10000000)
df2 = pd.read_csv(os.path.join(rootpath, "data/log_standard_4_22_to_5_08_pure.csv"), nrows=10000000)

user_features = pd.read_csv(os.path.join(rootpath, "data/user_features_pure.csv"))

video_features_basic = pd.read_csv(os.path.join(rootpath, "data/video_features_basic_pure.csv"), nrows=10000000)
video_features_basic["tag"] = video_features_basic["tag"].map(lambda x: eval(x + ",") if x is not np.nan else ())
video_features_statistics = pd.read_csv(os.path.join(rootpath, "data/video_features_statistic_pure.csv"), nrows=10000000) 
# 这个表是一些带有时效性的统计特征，例如近一个月的点击率等

# print("===================================================")
# print("The random data in 'log_random_4_22_to_5_08_1k.csv'")
# print("---------------------------------------------------")
# print(df_rand)


# print("===================================================")
# print("The standard data in 'log_standard_XXX.csv'")
# print("---------------------------------------------------")
# print(df1)

# print("===================================================")
# print("The user features in 'user_features_1k.csv'")
# print("---------------------------------------------------")
# print(user_features)

# print("===================================================")
# print("The basic video features in 'video_features_basic.csv'")
# print("---------------------------------------------------")
# print(video_features_basic)

# print("===================================================")
# print("The statistical features of videos in 'video_features_statistic_1k.csv'")
# print("---------------------------------------------------")
# print(video_features_statistics)

print("All data loaded.")
# ipdb.set_trace()


# num_feat = video_features_basic["tag"].map(len) # 数每一行有几个tag
# print(num_feat.describe())
# visual_statistics_discrete(num_feat, "视频标签数目")


cnt = video_features_basic["tag"].map(collections.Counter) # 对每一行，tag列的值变为{tag:count}，即具体标签和出现次数的字典
cnt_all = collections.Counter()
for d in cnt: # 这样就得到所有rows总的标签情况
    cnt_all.update(d) # counter的update和dict的不同，这里的value会加起来
# print(dict(cnt_all))
all_feat = pd.Series(sorted(list(itertools.chain.from_iterable([[i]*k for i,k in cnt_all.items()]))),name="feat")
# 上一行将“item-single_tag”的关系排开，组成新的Series
# print(all_feat)
visual_statistics_discrete(all_feat, "标签", size=(12,4.5), rotation=30, interval=0.05)
# ipdb.set_trace()
a = 1

# df_train = df1.append(df2) # append方法在现版本的pd中已经弃用了，ignore_index默认为false，则index会重复
df_train = df1._append(df2)
df_train = df_train.reset_index(drop=True) 
# 会重新用自然数为df做index，默认drop为false，则原有的index会变成一列，drop为true时，原有的index会被直接丢弃
df_train["domain"] = "Standard"
df_rand["domain"] = "Random"
df_all = df_train._append(df_rand)
df_all = df_all.reset_index(drop=True)
df_all_with_tag = df_all.merge(video_features_basic,'left', on='video_id')
# visual_with_hue(df_all_with_tag, var='标签分布', x='tag', hue='domain') #会报错因为有的video有多个tags

# %%
# duration_train = df_train["duration_ms"] / 1000
# print(duration_train.describe())
# visual_continue(duration_train, var="训练集视频时长（秒）")

# play_train = df_train["play_time_ms"] / 1000
# print(play_train.describe())
# visual_continue(play_train, var="训练集播放时长（秒）")

# maxx = 5
# ratio_train = df_train["play_time_ms"] / df_train["duration_ms"]
# ratio_train.loc[ratio_train.map(lambda x: x == np.inf)] = maxx
# ratio_train.loc[ratio_train > maxx] = maxx
# # 将播放比例大于5的都归为5

# print(ratio_train.describe())
# visual_continue(ratio_train, var="训练集播放比例")

# %%
# duration_rand = df_rand["duration_ms"] / 1000
# print(duration_rand.describe())
# visual_continue(duration_rand, var="测试集视频时长（秒）")

# play_rand = df_rand["play_time_ms"] / 1000
# print(play_rand.describe())
# visual_continue(play_rand, var="测试集播放时长（秒）")

# maxx = 5
# ratio_rand = df_rand["play_time_ms"] / df_rand["duration_ms"]
# ratio_rand.loc[ratio_rand.map(lambda x: x == np.inf)] = maxx
# ratio_rand.loc[ratio_rand > maxx] = maxx

# print(ratio_rand.describe())
# visual_continue(ratio_rand, var="测试集播放比例")

# %%


duration_all = df_all["duration_ms"] / 1000
play_all = df_all["play_time_ms"] / 1000
maxx = 5
ratio_all = df_all["play_time_ms"] / df_all["duration_ms"]
ratio_all.loc[ratio_all.map(lambda x: x == np.inf)] = maxx
ratio_all.loc[ratio_all > maxx] = maxx

df_all["duration_s"] = duration_all
df_all["play_time_s"] = play_all
df_all["watch_ratio"] = ratio_all


print(df_all.loc[df_all["domain"]=="Standard","play_time_s"].describe(),
      df_all.loc[df_all["domain"]=="Random","play_time_s"].describe())
print(df_all.loc[df_all["domain"]=="Standard","watch_ratio"].describe(),
      df_all.loc[df_all["domain"]=="Random","watch_ratio"].describe())

visual_with_hue(df_all, var="视频时长", x="duration_s", hue="domain", bin=100, is_sort=False, xrotation=0)
visual_with_hue(df_all, var="播放时长", x="play_time_s", hue="domain", bin=100, is_sort=False, xrotation=0)
visual_with_hue(df_all, var="观看比例", x="watch_ratio", hue="domain", bin=100, is_sort=False, xrotation=0)


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
df_rand.rename(columns={"video_id":"item_id"}, inplace=True)
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






