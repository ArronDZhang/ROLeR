# -*- coding: utf-8 -*-
# @Time    : 2023/1/18 17:20
# @Author  : Chongming GAO
# @FileName: visual_main_figure.py

import os
import re
import ipdb
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

from visual_utils import walk_paths, organize_df, loaddata, create_dir, handle_table


def axis_shift(ax1, x_shift=0.01, y_shift=0):
    position = ax1.get_position().get_points()
    pos_new = position
    pos_new[:, 0] += x_shift
    pos_new[:, 1] += y_shift
    ax1.set_position(Bbox(pos_new))


def compute_improvement(df, col, last=0):
    our = df.iloc[-5:][col]["CIRS"].mean()
    prev = df.iloc[-last:][col]["CIRS w_o CI"].mean()
    print(f"Improvement on [{col}] of last [{last}] count is {(our - prev) / prev}")


def visual4(df1, df2, save_fig_dir, savename="three"):
    visual_cols = ['R_tra', 'len_tra', 'ctr']

    # df1 = df1.iloc[:100]
    # df2 = df2.iloc[:200]
    # df3 = df3.iloc[:200]
    # df4 = df4.iloc[:1000]

    dfs = [df1, df2]
    series = "ABCD"
    dataset = ["KuaiRec", "KuaiRand"]
    # maxlen = [50, 100, 10, 30]
    fontsize = 11.5

    # all_method = sorted(set(df1['R_tra'].columns.to_list() +
    #                         df2['R_tra'].columns.to_list()))
    # methods_list = list(all_method)
    methods_list = ['DORL', 'MOPO', 'MBPO', 'IPS', 'BCQ', 'CQL', 'CRR', 'SQN', "UCB", r'$\epsilon$-greedy']

    num_methods = len(methods_list)

    colors = sns.color_palette(n_colors=num_methods)
    markers = ["o", "s", "p", "P", "X", "*", "h", "D", "v", "^", ">", "<", "x", "H"][:num_methods]

    color_kv = dict(zip(methods_list, colors))
    marker_kv = dict(zip(methods_list, markers))

    methods_list = methods_list[::-1]
    methods_order = dict(zip(methods_list, list(range(len(methods_list)))))

    fig = plt.figure(figsize=(5, 6))
    # plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3)
    axs = []
    for index in range(len(dfs)):
        alpha = series[index] # ABCD
        cnt = 1
        df = dfs[index]
        df.sort_index(axis=1, key=lambda col: [methods_order[x] for x in col.to_list()], level=1, inplace=True)
        ## 对‘列’排序，带key时，先对排序的依据进行lambda的处理之后再排序，但是不改变结果dataframe中的index
        ## 即lambda只在排序时起作用；所谓排序的依据，就比如说这里column中的level1，即方法名
        data_r = df[visual_cols[0]] # 取出一个数据集上所有方法在cumulative reward指标上的表现
        data_len = df[visual_cols[1]] # …在trajectory length指标上的表现
        data_ctr = df[visual_cols[2]] # …在immediate reward指标上的表现


        color = [color_kv[name] for name in data_r.columns]
        marker = [marker_kv[name] for name in data_r.columns]

        ax1 = plt.subplot2grid((3, 2), (0, index))
        data_r.plot(kind="line", linewidth=1, ax=ax1, legend=None, color=color, markevery=int(len(data_r) / 10),
                    fillstyle='none', alpha=.8, markersize=3) # 用data_r在画
        for i, line in enumerate(ax1.get_lines()):
            line.set_marker(marker[i])
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.grid(linestyle='dashdot', linewidth=0.8)
        ax1.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, loc="left", x=0.4, y=.97)
        ax1.set_title("{}".format(dataset[index]), fontsize=fontsize, y=1.1, fontweight=400)
        cnt += 1

        ax2 = plt.subplot2grid((3, 2), (1, index))
        data_len.plot(kind="line", linewidth=1, ax=ax2, legend=None, color=color, markevery=int(len(data_r) / 10),
                      fillstyle='none', alpha=.8, markersize=3) # 用data_len在画
        for i, line in enumerate(ax2.get_lines()):
            line.set_marker(marker[i])
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.grid(linestyle='dashdot', linewidth=0.8)
        ax2.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, y=.97)
        # ax2.set_title(r"$\it{Max round=" + str(maxlen[index]) + r"}$", fontsize=fontsize - 1.5, loc="left", x=-0.2,
        #               y=.97)
        cnt += 1

        ax3 = plt.subplot2grid((3, 2), (2, index))
        data_ctr.plot(kind="line", linewidth=1, ax=ax3, legend=None, color=color, markevery=int(len(data_r) / 10),
                      fillstyle='none', alpha=.8, markersize=3) # 用data_ctr在画
        for i, line in enumerate(ax3.get_lines()):
            line.set_marker(marker[i])
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        ax3.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, y=.97)
        ax3.set_xlabel("epoch", fontsize=11)
        cnt += 1
        plt.grid(linestyle='dashdot', linewidth=0.8)
        if index == 2:
            axis_shift(ax1, .015)
            axis_shift(ax2, .015)
            axis_shift(ax3, .015)
        if index == 3:
            axis_shift(ax1, .005)
            axis_shift(ax2, .005)
            axis_shift(ax3, .005)
        axs.append((ax1, ax2, ax3))

    ax1, ax2, ax3 = axs[0]
    ax1.set_ylabel("Cumulative reward", fontsize=9, fontweight=400)
    ax2.set_ylabel("Interaction length", fontsize=9, fontweight=400)
    ax3.set_ylabel("Single-round reward", fontsize=9, fontweight=400)
    # ax3.yaxis.set_label_coords(-0.17, 0.5)

    ax4 = axs[1][0]

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    dict_label = dict(zip(labels1, lines1))
    dict_label.update(dict(zip(labels2, lines2)))
    # dict_label = OrderedDict(sorted(dict_label.items(), key=lambda x: x[0]))
    methods_list = methods_list[::-1]
    dict_label = {k: dict_label[k] for k in methods_list}
    ax1.legend(handles=dict_label.values(), labels=dict_label.keys(), ncol=5,
               loc='lower left', columnspacing=0.7,
               bbox_to_anchor=(-0.28, 1.26), fontsize=9.5)

    # axo = plt.axes([0, 0, 1, 1], facecolor=(1, 1, 1, 0))
    # x, y = np.array([[0.505, 0.505], [0.06, 0.92]])
    # line = Line2D(x, y, lw=3, linestyle="dotted", color=(0.5, 0.5, 0.5))
    # axo.add_line(line)
    # plt.text(0.16, 0.02, "(A-B) Results with large interaction rounds", fontsize=11, fontweight=400)
    # plt.text(0.58, 0.02, "(C-D) Results with limited interaction rounds", fontsize=11, fontweight=400)
    # plt.axis('off')

    fig.savefig(os.path.join(save_fig_dir, savename + '.pdf'), format='pdf', bbox_inches='tight')
    # fig.savefig(os.path.join(save_fig_dir, savename + '.png'), format='png', bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print("done!")

def remove_redundent(df, level=1): # 这个函数是在去除方法后的‘leave’而非去除重复的方法
    methods = df.columns.levels[level]
    pattern_name = re.compile("(.*)[-\s]leave[=]?(\d+)")
    # methods = [pattern_name.match(method).group(1) for method in methods if pattern_name.match(method)]
    df.rename(columns={method:pattern_name.match(method).group(1) for method in methods if pattern_name.match(method)},
               level=level, inplace=True) # 把"method-leave=10" or "method leave 10"变成method
    df.rename(columns={"Ours": "DORL"}, level=level, inplace=True)

def to_latex(df, save_fig_dir, savename):
    df_latex, df_excel, df_avg = handle_table(df)

    df_latex1 = df_latex[["Free", "No Overlapping"]]

    filepath_latex = os.path.join(save_fig_dir, f"{savename}_table.tex")
    with open(filepath_latex, "w") as file:
        file.write(df_latex1.to_latex(escape=False))

    excel_path = os.path.join(save_fig_dir, savename + '.xlsx')
    df_excel.to_excel(excel_path)

def combile_two_tables(df1, df2, used_way, save_fig_dir, savename="main_result"): # 这个函数只是用来存结果整理后的LaTeX表格
    datasets = ["KuaiRec", "KuaiRand"]
    metrics = [r"$\text{R}_\text{tra}$", r"$\text{R}_\text{each}$", "Length", "MCD"]
    methods = ['DORL', 'MOPO', 'IPS', 'MBPO', 'BCQ', 'CQL', 'CRR', 'SQN', r'$\epsilon$-greedy', "UCB"][::-1]
    indices = [datasets, metrics]
    # df_all = pd.DataFrame(columns=pd.MultiIndex.from_product(indices, names=["Datasets", "Metrics", "Methods"]))
    df_all = pd.DataFrame(columns=pd.MultiIndex.from_product(indices))

    df_latex1, df_excel1, df_avg1 = handle_table(df1)
    df_latex2, df_excel2, df_avg2 = handle_table(df2)

    df_all["KuaiRec"] = df_latex1[used_way]
    df_all["KuaiRand"] = df_latex2[used_way]

    methods_order = dict(zip(methods, list(range(len(methods)))))
    df_all.sort_index(key=lambda index: [methods_order[x] for x in index.to_list()], inplace=True)


    filepath_latex = os.path.join(save_fig_dir, f"{savename}_table.tex")
    with open(filepath_latex, "w") as file:
        file.write(df_all.to_latex(escape=False))

    # excel_path = os.path.join(save_fig_dir, savename + '.xlsx')
    # df_excel.to_excel(excel_path)
    print("latex tex file produced!")

def visual_two_groups():
    realpath = os.path.dirname(__file__)
    save_fig_dir = os.path.join(realpath, "figures")

    create_dirs = [save_fig_dir]
    create_dir(create_dirs)

    dirpath = "./results_all"

    ways = {'FB', 'NX_0_', 'NX_10_'}
    metrics = {'ctr', 'len_tra', 'R_tra', 'ifeat_feat'}

    result_dir1 = os.path.join(dirpath, "kuairec")
    filenames = walk_paths(result_dir1)
    dfs1 = loaddata(result_dir1, filenames)
    
    df1 = organize_df(dfs1, ways, metrics)

    result_dir2 = os.path.join(dirpath, "kuairand")
    filenames = walk_paths(result_dir2)
    dfs2 = loaddata(result_dir2, filenames)
    df2 = organize_df(dfs2, ways, metrics)

    remove_redundent(df1, level=2) 
    remove_redundent(df2, level=2)

    savename = "main_result"


    # to_latex(df1, save_fig_dir, "kuairec")
    # to_latex(df2, save_fig_dir, "kuairand")


    way = "NX_0_" # 只取一种方式即No Overlapping
    df1_one, df2_one = df1[way], df2[way]


    visual4(df1_one, df2_one, save_fig_dir, savename=savename)

#    combile_two_tables(df1, df2, used_way="No Overlapping", save_fig_dir=save_fig_dir, savename=savename)


if __name__ == '__main__':
    visual_two_groups()