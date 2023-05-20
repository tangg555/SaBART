"""
@Desc:
@Reference:
bar plot
https://www.cnblogs.com/pyhou/p/12535208.html
line plot
https://blog.csdn.net/Treasure99/article/details/106044114
@Notes:
"""

import sys
import numpy as np
from pathlib import Path
import json

import pandas
from tqdm import tqdm
import shutil
import os
import json

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from transformers import BartTokenizer
from tqdm import tqdm
from collections import Counter
from src.utils import nlg_eval_utils

import pandas as pd  # pandas是一个强大的分析结构化数据的工具集
import numpy as np  # numpy是Python中科学计算的核心库
import matplotlib.pyplot as plt  # matplotlib数据可视化神器
from analyze import file_to_list

# 正态分布的概率密度函数
#   x      数据集中的某一具体测量值
#   mu     数据集的平均值，反映测量值分布的集中趋势
#   sigma  数据集的标准差，反映测量值分布的分散程度
def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf


def get_sample_groups(gen_result_dir: Path, group_num=10):
    triple_nums = file_to_list(gen_result_dir.joinpath("triple_nums.txt"))
    sample_triple_num_pairs = [(idx, int(one)) for idx, one in enumerate(triple_nums)]
    sample_triple_num_pairs.sort(key=lambda x: x[1])
    length = len(sample_triple_num_pairs)
    scale = int(length / group_num)
    data_groups = []
    for slice in range(group_num):
        data_groups.append(sample_triple_num_pairs[slice*scale:(slice+1)*scale])
    return data_groups

def plot_data_groups(data_groups):
    bar_data = [np.mean([pair[1] for pair in group]) for group in data_groups]
    labels = [str(idx) for idx in range(len(bar_data))]

    graph_font = {'family': 'serif',
            'style': 'normal',
            'weight': 'bold',
            'color': 'black',
            'size': 14
            }

    plt.title('Avg. Triples Per Group', fontdict=graph_font)
    plt.xlabel('Test Group', fontdict=graph_font)
    plt.ylabel('Avg. Triples', fontdict=graph_font)

    plt.bar(range(len(bar_data)), bar_data, tick_label=labels, align='center',color ='orange',alpha=0.8, zorder=10)

    x = list(range(len(bar_data)))

    text_font = {'family': 'serif',
            'style': 'normal',
            'weight': 'bold',
            'color': 'black',
            'size': 10
            }
    for a, b in zip(x, bar_data):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontdict=text_font)
    plt.grid(zorder=0)  # 画网格
    plt.show()

def plot_dist_of_triples(gen_result_path: Path):
    triple_nums = file_to_list(gen_result_path.joinpath("triple_nums.txt"))
    triple_nums = [int(one) for one in triple_nums]
    min_ = min(triple_nums)
    max_ = max(triple_nums)
    data = pandas.Series(triple_nums)  # 获得长度数据集
    mean = data.mean()  # 获得数据集的平均值
    std = data.std()  # 获得数据集的标准差

    # 设定X轴：前两个数字是X轴的起止范围，第三个数字表示步长
    # 步长设定得越小，画出来的正态分布曲线越平滑
    x = np.arange(min_, max_, 0.1)
    # 设定Y轴，载入刚才定义的正态分布函数
    y = normfun(x, mean, std)
    # 绘制数据集的正态分布曲线
    plt.plot(x, y)
    plt.grid()

    graph_font = {'family': 'serif',
            'style': 'normal',
            'weight': 'bold',
            'color': 'black',
            'size': 14
            }
    # 绘制数据集的直方图
    plt.hist(data, bins=20, rwidth=0.9, density=True)
    plt.title('Knowledge Density distribution', fontdict=graph_font)
    plt.xlabel('Triples Per Sample', fontdict=graph_font)
    plt.ylabel('Probability Distribution', fontdict=graph_font)

    # 输出正态分布曲线和直方图
    plt.show()

def calculate_es_score(response, post_entities: list):
    score = 0
    for ent in post_entities:
        if ent in response:
            score += 1
    return score

def group_es_score(group_ids, all_response, all_ents):
    scores = []
    for idx in group_ids:
        scores.append(calculate_es_score(all_response[idx], all_ents[idx]))
    return np.mean(scores)

def plot_es_score_curve(gen_result_dir: Path, data_groups):
    testset_responses = file_to_list(gen_result_dir.joinpath("testset_responses.txt"))
    chatbot_onehop_gen = file_to_list(gen_result_dir.joinpath("chatbot_onehop_gen.txt"))
    chatbot_onehop_no_kgagg_gen = file_to_list(gen_result_dir.joinpath("chatbot_onehop_no_agg_gen.txt"))
    chatbot_onehop_no_child_gen = file_to_list(gen_result_dir.joinpath("chatbot_onehop_no_child_gen.txt"))
    conceptflow_gen = file_to_list(gen_result_dir.joinpath("conceptflow_gen.txt"))

    post_entities = []
    with gen_result_dir.joinpath("post_entities.txt").open("r", encoding="utf-8") as fr:
        for line in fr:
            post_entities.append(eval(line.strip()))

    scores = {
        # "golden": [],
              "SaBART": [],
              "- w/o dy-agg": [],
              "- w/o st-agg": [],
              "ConceptFlow": [],
              }
    for group in data_groups:
        group_ids = [pair[0] for pair in group]
        # scores["golden"].append(group_es_score(group_ids, testset_responses, post_entities))
        scores["SaBART"].append(group_es_score(group_ids, chatbot_onehop_gen, post_entities))
        scores["- w/o dy-agg"].append(group_es_score(group_ids, chatbot_onehop_no_kgagg_gen, post_entities))
        scores["- w/o st-agg"].append(group_es_score(group_ids, chatbot_onehop_no_child_gen, post_entities))
        scores["ConceptFlow"].append(group_es_score(group_ids, conceptflow_gen, post_entities))

    x = list(range(len(data_groups)))

    keys = list(scores.keys())
    plt.plot(x, scores[keys[0]], '^k-', label=keys[0])  # blue markers with default shape
    plt.plot(x, scores[keys[1]], 'or-', label=keys[1]) 	# red circles
    plt.plot(x, scores[keys[2]], 'db-', label=keys[2]) 	# red circles
    plt.plot(x, scores[keys[3]], 'xy-', label=keys[3]) 	# red circles
    # plt.plot(x, scores[keys[4]], 'Dg-', label=keys[4]) 	# red circles
    # plt.plot(x, scores[keys[5]], 'hc-', label=keys[5]) 	# red circles
    # plt.plot(x, scores[keys[6]], '+m-', label=keys[6]) 	# red circles
    # plt.plot(x, scores[keys[7]], '*navy-', label=keys[7]) 	# red circles

    graph_font = {'family': 'serif',
            'style': 'normal',
            'weight': 'bold',
            'color': 'black',
            'size': 14
            }

    plt.title('External Knowledge Incorporation', fontdict=graph_font)
    plt.xticks(x, rotation=0, fontproperties='serif', fontweight="bold", size=8)
    plt.xlabel('Test Group', fontdict=graph_font)
    plt.ylabel('Entities in Post', fontdict=graph_font)
    plt.legend(fontsize='x-large')
    plt.grid(zorder=0)  # 画网格
    plt.show()

def plot_es_score_bar(gen_result_dir: Path):
    testset_responses = file_to_list(gen_result_dir.joinpath("testset_responses.txt"))
    chatbot_onehop_gen = file_to_list(gen_result_dir.joinpath("chatbot_onehop_gen.txt"))
    chatbot_onehop_no_kgagg_gen = file_to_list(gen_result_dir.joinpath("chatbot_onehop_no_agg_gen.txt"))
    chatbot_onehop_no_child_gen = file_to_list(gen_result_dir.joinpath("chatbot_onehop_no_child_gen.txt"))
    mybart_gen = file_to_list(gen_result_dir.joinpath("mybart_gen.txt"))
    conceptflow_gen = file_to_list(gen_result_dir.joinpath("conceptflow_gen.txt"))

    post_entities = []
    with gen_result_dir.joinpath("post_entities.txt").open("r", encoding="utf-8") as fr:
        for line in fr:
            post_entities.append(eval(line.strip()))

    all_ids = list(range(len(testset_responses)))
    scores = {
        # "golden": group_es_score(all_ids, testset_responses, post_entities),
              "SaBart": group_es_score(all_ids, chatbot_onehop_gen, post_entities),
              "- w/o dy-agg": group_es_score(all_ids, chatbot_onehop_no_kgagg_gen, post_entities),
              "- w/o st-agg": group_es_score(all_ids, chatbot_onehop_no_child_gen, post_entities),
              "ConceptFlow": group_es_score(all_ids, conceptflow_gen, post_entities),
              }
    graph_font = {'family': 'serif',
            'style': 'normal',
            'weight': 'bold',
            'color': 'black',
            'size': 14
            }

    plt.title('Overall Knowledge Incorporation', fontdict=graph_font)
    plt.xlabel('Models', fontdict=graph_font)
    plt.ylabel('Entities in Post', fontdict=graph_font)

    x = list(range(len(scores.values())))

    plt.bar(x, list(scores.values()), tick_label=list(scores.keys()),
            align='center',
            color=list("krby"),
            hatch=["//", "//", "//", "//"],
            alpha=0.8, zorder=10)
    plt.xticks(rotation=0, fontproperties='serif', fontweight="bold", size=12)

    text_font = {'family': 'serif',
            'style': 'normal',
            'weight': 'bold',
            'color': 'black',
            'size': 10
            }
    for a, b in zip(x, scores.values()):
        plt.text(a, b + 0.01, round(b, 2), ha='center', va='bottom', fontdict=text_font)
    plt.grid(zorder=0)  # 画网格
    plt.show()

def get_metrics(gen_result_dir):
    testset_responses = file_to_list(gen_result_dir.joinpath("testset_responses.txt"))
    chatbot_onehop_gen = file_to_list(gen_result_dir.joinpath("chatbot_onehop_gen.txt"))
    chatbot_onehop_no_kgagg_gen = file_to_list(gen_result_dir.joinpath("chatbot_onehop_no_agg_gen.txt"))
    chatbot_onehop_no_child_gen = file_to_list(gen_result_dir.joinpath("chatbot_onehop_no_child_gen.txt"))
    mybart_gen = file_to_list(gen_result_dir.joinpath("mybart_gen.txt"))
    conceptflow_gen = file_to_list(gen_result_dir.joinpath("conceptflow_gen.txt"))
    print(f"chatbot_onehop_gen: {nlg_eval_utils.repetition_distinct_metric(chatbot_onehop_gen)}")
    print(f"chatbot_onehop_no_kgagg_gen: {nlg_eval_utils.repetition_distinct_metric(chatbot_onehop_no_kgagg_gen)}")
    print(f"chatbot_onehop_no_child_gen: {nlg_eval_utils.repetition_distinct_metric(chatbot_onehop_no_child_gen)}")
    print(f"mybart_gen: {nlg_eval_utils.repetition_distinct_metric(mybart_gen)}")

if __name__ == '__main__':
    gen_result_dir = Path(f"{BASE_DIR}/resources/generation_results")
    # 1. 画数据的分布图 --------------
    plot_dist_of_triples(gen_result_dir)

    data_groups = get_sample_groups(gen_result_dir, group_num=10)
    # # 2. 画分组数据的triples个数 --------------
    # plot_data_groups(data_groups)
    # # 3. 画分组数据的entity selection score 的曲线图 --------------
    # plot_es_score_curve(gen_result_dir, data_groups)

    # # 4. 画entity selection score 的 整个的柱状图--------------
    # plot_es_score_bar(gen_result_dir)

    # # 5. get metrics--------------
    # get_metrics(gen_result_dir)
