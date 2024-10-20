import os
import re
import json
import pandas as pd
import seaborn as sns
from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def create_dir(create_dirs):
    """
    Create the required directories.
    """
    for dir in create_dirs:
        if not os.path.exists(dir):
            # logger.info('Create dir: %s' % dir)
            try:
                os.mkdir(dir)
            except FileExistsError:
                print("The dir [{}] already existed".format(dir))

def walk_paths(result_dir):
    g = os.walk(result_dir)
    # print(f"Reading all logs under [{result_dir}]")
    files = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if file_name[0] == '.' or file_name[0] == '_':
                continue
            # print(os.path.join(path, file_name))
            files.append(file_name)
    return files


def organize_df(dfs, ways, metrics):
    indices = [list(dfs.keys()), ways, metrics]

    df_all = pd.DataFrame(columns=pd.MultiIndex.from_product(indices, names=["Exp", "ways", "metrics"]))

    for message, df in dfs.items():
        for way in ways:
            for metric in metrics:
                col = (way if way != "FB" else "") + metric
                df_all[message, way, metric] = df[col]

    # # Rename MultiIndex columns in Pandas
    # # https://stackoverflow.com/questions/41221079/rename-multiindex-columns-in-pandas
    # df_all.rename(
    #     columns={"RL_val_trajectory_reward": "R_tra", "RL_val_trajectory_len": 'len_tra', "RL_val_CTR": 'ctr'},
    #     level=1,inplace=True)

    # change order of levels
    # https://stackoverflow.com/questions/29859296/how-do-i-change-order-grouping-level-of-pandas-multiindex-columns

    df_all.columns = df_all.columns.swaplevel(0, 2)
    df_all.sort_index(axis=1, level=0, inplace=True)
    df_all.columns = df_all.columns.swaplevel(0, 1)

    all_method = set(df_all.columns.levels[2].to_list())
    all_method_map = {}
    for method in all_method:
        res = re.match("\[([KT]_)?(.+?)(_len.+)?\]", method)
        if res:
            all_method_map[method] = res.group(2)

    df_all.rename(
        columns=all_method_map,
        level=2, inplace=True)

    df_all.rename(
        columns={"CIRSwoCI": 'CIRS w/o CI',
                 "epsilon-greedy": r'$\epsilon$-greedy',
                 "DeepFM+Softmax": 'DeepFM'},
        level=2, inplace=True)

    return df_all


def loaddata(dirpath, filenames, use_filename=True):
    pattern_epoch = re.compile("Epoch: \[(\d+)]")
    pattern_info = re.compile("Info: \[(\{.+\})]")
    pattern_message = re.compile('"message": "(.+)"')
    pattern_array = re.compile("array\((.+?)\)")

    pattern_tau = re.compile('"tau": (.+),')
    pattern_read = re.compile('"read_message": "(.+)"')

    dfs = {}
    infos = {}

    for filename in filenames:
        if filename[0] == '.' or filename[0] == '_':  # ".DS_Store":
            continue
        df = pd.DataFrame()
        message = "None"
        filepath = os.path.join(dirpath, filename)
        with open(filepath, "r") as file:
            lines = file.readlines()
            start = False
            add = 0
            info_extra = {'tau': 0, 'read': ""}
            for i, line in enumerate(lines):
                res_tau = re.search(pattern_tau, line)
                if res_tau:
                    info_extra['tau'] = res_tau.group(1)
                res_read = re.search(pattern_read, line)
                if res_read:
                    info_extra['read'] = res_read.group(1)

                res = re.search(pattern_epoch, line)
                if res:
                    epoch = int(res.group(1))
                    if (start == False) and epoch == 0:
                        add = 1
                        start = True
                    epoch += add
                    info = re.search(pattern_info, line)
                    try:
                        info1 = info.group(1).replace("\'", "\"")
                    except Exception as e:
                        print("jump incomplete line: [{}]".format(line))
                        continue
                    info2 = re.sub(pattern_array, lambda x: x.group(1), info1)

                    data = json.loads(info2)
                    df_data = pd.DataFrame(data, index=[epoch], dtype=float)
                    # df = df.append(df_data)
                    df = pd.concat([df, df_data])
                res_message = re.search(pattern_message, line)
                if res_message:
                    message = res_message.group(1)

            if use_filename:
                message = filename[:-4]

            # print(file.name)
            df.rename(
                columns={"RL_val_trajectory_reward": "R_tra",
                         "RL_val_trajectory_len": 'len_tra',
                         "RL_val_CTR": 'ctr'},
                inplace=True)

            df.rename(
                columns={"trajectory_reward": "R_tra",
                         "trajectory_len": 'len_tra',
                         "CTR": 'ctr'},
                inplace=True)

        dfs[message] = df
        infos[message] = info_extra

    dfs = OrderedDict(sorted(dfs.items(), key=lambda item: len(item[1]), reverse=True))
    return dfs


def get_top2_methods(col, is_largest):
    if is_largest:
        top2_name = col.nlargest(2).index.tolist()
    else:
        top2_name = col.nsmallest(2).index.tolist()
    name1, name2 = top2_name[0], top2_name[1]
    return name1, name2

def handle_one_col(df_metric, final_rate, is_largest):
    length = len(df_metric)
    res_start = int((1 - final_rate) * length)
    mean = df_metric[res_start:].mean()
    std = df_metric[res_start:].std()

    # mean.nlargest(2).index[1]
    res_latex = pd.Series(map(lambda mean, std: f"${mean:.4f}\pm {std:.4f}$", mean, std),
                          index=mean.index)
    res_excel = pd.Series(map(lambda mean, std: f"{mean:.4f}+{std:.4f}", mean, std),
                          index=mean.index)
    res_avg = mean

    name1, name2 = get_top2_methods(mean, is_largest=is_largest)
    res_latex.loc[name1] = r"$\mathbf{" + r"{}".format(res_latex.loc[name1][1:-1]) + r"}$"
    res_latex.loc[name2] = r"\underline{" + res_latex.loc[name2] + r"}"

    return res_latex, res_excel, res_avg

def handle_table(df_all, final_rate=1, methods=['DORL', 'CIRS', 'MOPO', 'MBPO', 'IPS', 'BCQ', 'CQL', 'CRR', 'SQN', r'$\epsilon$-greedy', "UCB", "ROLeR"]):
    df_all.rename(columns={"FB": "Free", "NX_0_": r"No Overlapping", "NX_10_": r"No Overlapping with 10 turns"},
                  level=0, inplace=True)
    df_all.rename(columns={"R_tra": r"$\text{R}_\text{tra}$", "ifeat_feat": "MCD",
                           "CV_turn": r"$\text{CV}_\text{M}$", "len_tra": "Length",
                           "ctr": r"$\text{R}_\text{each}$"}, level=1,
                  inplace=True)
    df_all.rename(columns={"epsilon-greedy": r'$\epsilon$-greedy'}, inplace=True)

    ways = df_all.columns.levels[0][::-1]
    metrics = df_all.columns.levels[1]
    if methods is None:
        methods = df_all.columns.levels[2].to_list()

    methods_order = dict(zip(methods, list(range(len(methods)))))

    df_latex = pd.DataFrame(columns=pd.MultiIndex.from_product([ways, metrics], names=["ways", "metrics"]))
    df_excel = pd.DataFrame(columns=pd.MultiIndex.from_product([ways, metrics], names=["ways", "metrics"]))
    df_avg = pd.DataFrame(columns=pd.MultiIndex.from_product([ways, metrics], names=["ways", "metrics"]))

    for col, way in enumerate(ways):
        df = df_all[way]
        for row, metric in enumerate(metrics):
            df_metric = df[metric]
            is_largest = False if metric == "MCD" else True
            df_latex[way, metric], df_excel[way, metric], df_avg[way, metric] = handle_one_col(df_metric, final_rate, is_largest=is_largest)

    df_latex.sort_index(key=lambda index: [methods_order[x] for x in index.to_list()], inplace=True)
    df_excel.sort_index(key=lambda index: [methods_order[x] for x in index.to_list()], inplace=True)
    df_avg.sort_index(key=lambda index: [methods_order[x] for x in index.to_list()], inplace=True)

    # print(df_latex.to_markdown())
    # excel_path = os.path.join(save_fig_dir, savename + '.xlsx')
    # df_excel.to_excel(excel_path)

    return df_latex, df_excel, df_avg


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
    methods_list = ['CIRS', r'$\epsilon$-greedy', "UCB", 'SQN', 'CRR', 'CQL', 'BCQ', 'IPS', 'MBPO', 'MOPO', 'DORL', "ROLeR"]

    num_methods = len(methods_list)

    colors = sns.color_palette("husl", n_colors=10)
    colors.append("skyblue")
    colors.append("red")
    markers = ["o", "s", "p", "P", "X", "h", "D", "v", "^", ">", "<", "*", "x", "H"][:num_methods]

    color_kv = dict(zip(methods_list, colors))
    marker_kv = dict(zip(methods_list, markers))

    methods_list = methods_list[::-1]
    methods_order = dict(zip(methods_list, list(range(len(methods_list)))))

    fig = plt.figure(figsize=(10, 12))
    plt.subplots_adjust(wspace=0.2)
    plt.subplots_adjust(hspace=0.2)
    axs = []
    for index in range(len(dfs)):
        alpha = series[index]
        cnt = 1
        df = dfs[index]
        df.sort_index(axis=1, key=lambda col: [methods_order[x] for x in col.to_list()], level=1, inplace=True)

        data_r = df[visual_cols[0]]
        data_len = df[visual_cols[1]]
        data_ctr = df[visual_cols[2]]

        color = [color_kv[name] for name in data_r.columns]
        marker = [marker_kv[name] for name in data_r.columns]

        ax1 = plt.subplot2grid((3, 2), (0, index))
        data_r["ROLeR"].plot(kind="line", linewidth=1.5, ax=ax1, legend=None, color=color[0], markevery=int(len(data_r) / 15),
                    fillstyle='none', alpha=.9, markersize=10)
        data_r.drop("ROLeR", axis=1).plot(kind="line", linewidth=1, ax=ax1, legend=None, color=color[1:], markevery=int(len(data_r) / 15),
                    fillstyle='none', alpha=.7, markersize=5)
        for i, line in enumerate(ax1.get_lines()):
            line.set_marker(marker[i])
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.grid(linestyle='dashdot', linewidth=0.8)
        ax1.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, loc="left", x=0.45, y=1)
        ax1.set_title("{}".format(dataset[index]), fontsize=15, y=1.1, fontweight=400)
        cnt += 1

        ax2 = plt.subplot2grid((3, 2), (1, index))
        data_len["ROLeR"].plot(kind="line", linewidth=1.5, ax=ax2, legend=None, color=color[0], markevery=int(len(data_r) / 15),
                    fillstyle='none', alpha=.9, markersize=10)
        data_len.drop("ROLeR", axis=1).plot(kind="line", linewidth=1, ax=ax2, legend=None, color=color[1:], markevery=int(len(data_r) / 15),
                      fillstyle='none', alpha=.7, markersize=5)
        for i, line in enumerate(ax2.get_lines()):
            line.set_marker(marker[i])
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.grid(linestyle='dashdot', linewidth=0.8)
        ax2.set_title("({}{})".format(alpha, cnt), fontsize=fontsize)
        cnt += 1

        ax3 = plt.subplot2grid((3, 2), (2, index))
        data_ctr["ROLeR"].plot(kind="line", linewidth=1.5, ax=ax3, legend=None, color=color[0], markevery=int(len(data_r) / 15),
                    fillstyle='none', alpha=.9, markersize=10)
        data_ctr.drop("ROLeR", axis=1).plot(kind="line", linewidth=1, ax=ax3, legend=None, color=color[1:], markevery=int(len(data_r) / 15),
                      fillstyle='none', alpha=.7, markersize=5)
        for i, line in enumerate(ax3.get_lines()):
            line.set_marker(marker[i])
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        ax3.set_title("({}{})".format(alpha, cnt), fontsize=fontsize)
        ax3.set_xlabel("epoch", fontsize=15)
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
    ax1.set_ylabel("Cumulative reward", fontsize=15, fontweight=400)
    ax2.set_ylabel("Interaction length", fontsize=15, fontweight=400)
    ax3.set_ylabel("Single-round reward", fontsize=15, fontweight=400)

    ax4 = axs[1][0]

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    dict_label = dict(zip(labels1, lines1))
    dict_label.update(dict(zip(labels2, lines2)))
    # dict_label = OrderedDict(sorted(dict_label.items(), key=lambda x: x[0]))
    methods_list = methods_list[::-1]
    for k in methods_list:
        if k in dict_label:
            dict_label[k] = dict_label[k]
    #dict_label = {k: dict_label[k] for k in methods_list}
    ax1.legend(handles=dict_label.values(), labels=dict_label.keys(), ncol=6,
               loc='lower left', bbox_to_anchor=(0,1.2), columnspacing=1.2, fontsize=13)

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

def remove_redundent(df, level=1):
    methods = df.columns.levels[level]
    pattern_name = re.compile("(.*)[-\s]leave[=]?(\d+)")
    # methods = [pattern_name.match(method).group(1) for method in methods if pattern_name.match(method)]
    df.rename(columns={method:pattern_name.match(method).group(1) for method in methods if pattern_name.match(method)},
               level=level, inplace=True)
    df.rename(columns={"Ours": "DORL"}, level=level, inplace=True)

def to_latex(df, save_fig_dir, savename):
    df_latex, df_excel, df_avg = handle_table(df)

    df_latex1 = df_latex[["Free", "No Overlapping"]]

    filepath_latex = os.path.join(save_fig_dir, f"{savename}_table.tex")
    with open(filepath_latex, "w") as file:
        file.write(df_latex1.to_latex(escape=False))

    excel_path = os.path.join(save_fig_dir, savename + '.xlsx')
    df_excel.to_excel(excel_path)

def combile_two_tables(df1, df2, used_way, save_fig_dir, savename="main_result"):
    datasets = ["KuaiRec", "KuaiRand"]
    metrics = [r"$\text{R}_\text{tra}$", r"$\text{R}_\text{each}$", "Length", "MCD"]
    methods = ['DORL', 'CIRS', 'MOPO', 'IPS', 'MBPO', 'BCQ', 'CQL', 'CRR', 'SQN', r'$\epsilon$-greedy', "UCB", "ROLeR"][::-1]
    indices = [datasets, metrics]
    # df_all = pd.DataFrame(columns=pd.MultiIndex.from_product(indices, names=["Datasets", "Metrics", "Methods"]))
    df_all = pd.DataFrame(columns=pd.MultiIndex.from_product(indices))

    df_latex1, df_excel1, df_avg1 = handle_table(df1)
    df_latex2, df_excel2, df_avg2 = handle_table(df2)

    df_all["KuaiRec"] = df_latex1[used_way]
    df_all["KuaiRand"] = df_latex2[used_way]

    df_all.fillna("-", inplace = True)

    methods_order = dict(zip(methods, list(range(len(methods)))))
    df_all.sort_index(key=lambda index: [methods_order[x] for x in index.to_list()], inplace=True)

    filepath_latex = os.path.join(save_fig_dir, f"{savename}_table.tex")
    with open(filepath_latex, "w") as file:
        file.write(df_all.to_latex(escape=False))

    # excel_path = os.path.join(save_fig_dir, savename + '.xlsx')
    # df_excel.to_excel(excel_path)
    print("latex tex file produced!")

def visual_two_groups():
    realpath = os.getcwd()
    save_fig_dir = os.path.join(realpath, "figures")

    create_dirs = [save_fig_dir]
    create_dir(create_dirs)

    # dirpath = "./results_all"
    dirpath = "./table4"

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

    way = "NX_0_"
    df1_one, df2_one = df1[way], df2[way]

    visual4(df1_one, df2_one, save_fig_dir, savename=savename)

    combile_two_tables(df1, df2, used_way="No Overlapping", save_fig_dir=save_fig_dir, savename=savename)

if __name__ == '__main__':
    visual_two_groups()