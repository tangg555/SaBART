"""
@Desc:
@Reference:
t-SNE:
https://zhuanlan.zhihu.com/p/358195652
@Notes:
"""

import sys
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import shutil
import os
import json

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from transformers import BartTokenizer
from tqdm import tqdm
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import torch

from plot_metrics import file_to_list
from src.configuration.chatbot.config_args import parse_args_for_config
from src.utils.file_utils import save_json, load_json
from src.utils.misc_utils import reverse_dict_key_val
from src.models.chabot import (
    MyBart,
    OneHopGNNBart,
    OneHopGNNBartNoAgg,
    OneHopGNNBartNoChild,
    OneHopGNNBartEH,
)
import random
random.seed(42)
np.random.seed(42)

def run_examples():
    def get_data():
        digits = datasets.load_digits(n_class=6)
        data = digits.data
        label = digits.target
        n_samples, n_features = data.shape
        return data, label, n_samples, n_features

    def plot_embedding(data, label, title):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)

        fig = plt.figure()
        ax = plt.subplot(111)
        mid = int(data.shape[0] / 2)
        for i in range(mid):
            plt.text(data[i, 0], data[i, 1], str("o"),  # 此处的含义是用什么形状绘制当前的数据点
                     color=plt.cm.Set1(1 / 10),  # 表示颜色
                     fontdict={'weight': 'bold', 'size': 9})  # 字体和大小
        for i in range(mid):
            plt.text(data[mid + i, 0], data[mid + i, 1], str("o"),
                     color=plt.cm.Set1(2 / 10),
                     fontdict={'weight': 'bold', 'size': 9})

        plt.xticks([-0.1, 1.1])  # 坐标轴设置
        # xticks(locs, [labels], **kwargs)locs，是一个数组或者列表，表示范围和刻度，label表示每个刻度的标签
        plt.yticks([0, 1.1])
        plt.title(title)
        plt.show()

    def main():
        data, label, n_samples, n_features = get_data()
        print('Computing t-SNE embedding')
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        t0 = time()
        result = tsne.fit_transform(data)
        plot_embedding(result, label,
                       't-SNE embedding of the digits (time %.2fs)'
                       % (time() - t0))
    main()

def json_key_str_to_id(dict_obj: dict):
    new_dict = {}
    for key, val in dict_obj.items():
        new_dict[int(key)] = val
    return new_dict

def load_model_embeddings():
    print("Loading model embeddings -----")
    hparams = parse_args_for_config()
    model: OneHopGNNBart = OneHopGNNBart(hparams)
    embeddings = model.model.get_input_embeddings()
    id2word_file = Path(f"{BASE_DIR}/datasets/cadge/id2word.txt")
    id2word = json_key_str_to_id(load_json(id2word_file))
    embed_dict = {}
    for idx in range(len(id2word)):
        embed_dict[id2word[idx]] = embeddings(torch.tensor(idx)).detach().numpy().tolist()
    print("Done-----")
    return embed_dict

def load_gnn_embeddings():
    glove_file = Path(f"{BASE_DIR}/resources/commonsense_conversation_dataset/glove.840B.300d.txt")
    print("Loading Glove Model -----")
    glove_model = {}
    with glove_file.open("r", encoding="utf-8") as fr:
        try:
            for line in fr:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                glove_model[word] = embedding
        except Exception as e:
            print(f"wrong line: {line}")
    print(f"{len(glove_model)} words loaded!")
    print("Done-----")
    return glove_model

def get_data_from_samples(samples, entities, ent_embeds:dict, norm_embeds:dict):
    ent_data_ = []
    norm_data_ = []
    for post in samples:
        for word in post.strip().split():
            word = word.strip()
            if word in entities and word in ent_embeds:
                ent_data_.append(ent_embeds[word])
            elif f"Ġ{word}" in norm_embeds:
                norm_data_.append(norm_embeds[f"Ġ{word}"])
    ent_data_ = np.array(ent_data_, dtype=float)
    norm_data_ = np.array(norm_data_, dtype=float)
    return ent_data_, norm_data_


def get_post_entities():
    raw = file_to_list(gen_result_dir.joinpath("post_entities.txt"))
    entities = set()
    for one in raw:
        entities.update(eval(one))
    return entities

def normalize_data(ent_data, norm_data):
    data = np.concatenate((ent_data, norm_data), axis=0)
    min_v, max_v = np.min(data, 0), np.max(data, 0)
    ent_data = (ent_data - min_v) / (max_v - min_v)
    norm_data = (norm_data - min_v) / (max_v - min_v)
    return ent_data, norm_data

def scale_norm_data(data: np.array, scale= np.array([5, 5])):
    data =  (data - np.mean(data, axis=0)) * scale
    data += 0.8
    return data

def scale_ent_data(ent_data: np.array, scale=np.array([0.3, 0.3])):
    data = (ent_data - np.mean(ent_data, axis=0)) * scale
    data += 0.5
    return data

def generate_ents(entities, ent_embeds: dict, limit=50):
    ent_data_ = []
    ent_words = random.sample(entities, limit)
    for word in ent_words:
        if f"{word}" in ent_embeds:
            ent_data_.append(ent_embeds[f"{word}"])
    ent_data_ = np.array(ent_data_, dtype=float)
    return ent_data_

POST_LIMIT = 32

def plot_gnn_scatters(gen_result_dir: Path):
    testset_posts = file_to_list(gen_result_dir.joinpath("testset_posts.txt"))
    post_entities = get_post_entities()
    samples = testset_posts[:POST_LIMIT]
    gnn_embedds = load_gnn_embeddings()
    model_embedds = load_model_embeddings()
    ent_limit = 50
    norm_limit =200

    # normal gnn
    ent_data_, norm_data_ = get_data_from_samples(samples, post_entities, ent_embeds=gnn_embedds, norm_embeds=model_embedds)
    # random_ent_data = generate_ents(post_entities, gnn_embedds, limit=ent_limit)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    ent_data = tsne.fit_transform(ent_data_)
    norm_data = tsne.fit_transform(norm_data_)
    ent_data, norm_data = normalize_data(ent_data, norm_data)
    #  -------- scale --------
    # norm_data = scale_norm_data(norm_data[:norm_limit])
    # ent_data = scale_ent_data(ent_data[:ent_limit])
    print(f"print dot: norm_data: {len(norm_data)}; ent_data: {len(ent_data)}")
    for idx in range(len(ent_data)):
        # plt.text(ent_data[idx, 0], ent_data[idx, 1], str("o"),  # 此处的含义是用什么形状绘制当前的数据点
        #          color=plt.cm.Set1(1 / 10),  # 表示颜色
        #          fontdict={'weight': 'bold', 'size': 9})  # 字体和大小
        plt.scatter(ent_data[idx, 0], ent_data[idx, 1], s=10, color=plt.cm.Set1(1 / 10))
    for idx in range(len(norm_data)):
        # plt.text(norm_data[idx, 0], norm_data[idx, 1], str("o"),
        #          color=plt.cm.Set1(2 / 10),
        #          fontdict={'weight': 'bold', 'size': 9})
        plt.scatter(norm_data[idx, 0], norm_data[idx, 1], s=10, color=plt.cm.Set1(2 / 10))

    plt.xticks([-0.1, 1.1])  # 坐标轴设置
    # xticks(locs, [labels], **kwargs)locs，是一个数组或者列表，表示范围和刻度，label表示每个刻度的标签
    plt.yticks([-0.1, 1.1])
    plt.show()


def plot_lm_scatters(gen_result_dir: Path):
    testset_posts = file_to_list(gen_result_dir.joinpath("testset_posts.txt"))
    post_entities = get_post_entities()
    samples = testset_posts[:POST_LIMIT]
    model_embedds = load_model_embeddings()
    ent_limit = 50
    norm_limit =200

    # normal gnn
    ent_data_, norm_data_ = get_data_from_samples(samples, post_entities, ent_embeds=model_embedds, norm_embeds=model_embedds)
    # random_ent_data = generate_ents(post_entities, gnn_embedds, limit=ent_limit)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    ent_data = tsne.fit_transform(ent_data_)
    norm_data = tsne.fit_transform(norm_data_)
    ent_data, norm_data = normalize_data(ent_data, norm_data)
    #  -------- scale --------
    # norm_data = scale_norm_data(norm_data[:norm_limit])
    # ent_data = scale_ent_data(ent_data[:ent_limit])
    print(f"print dot: norm_data: {len(norm_data)}; ent_data: {len(ent_data)}")
    for idx in range(len(ent_data)):
        # plt.text(ent_data[idx, 0], ent_data[idx, 1], str("o"),  # 此处的含义是用什么形状绘制当前的数据点
        #          color=plt.cm.Set1(1 / 10),  # 表示颜色
        #          fontdict={'weight': 'bold', 'size': 9})  # 字体和大小
        plt.scatter(ent_data[idx, 0], ent_data[idx, 1], s=10, color=plt.cm.Set1(1 / 10))
    for idx in range(len(norm_data)):
        # plt.text(norm_data[idx, 0], norm_data[idx, 1], str("o"),
        #          color=plt.cm.Set1(2 / 10),
        #          fontdict={'weight': 'bold', 'size': 9})
        plt.scatter(norm_data[idx, 0], norm_data[idx, 1], s=10, color=plt.cm.Set1(2 / 10))

    plt.xticks([-0.1, 1.1])  # 坐标轴设置
    # xticks(locs, [labels], **kwargs)locs，是一个数组或者列表，表示范围和刻度，label表示每个刻度的标签
    plt.yticks([-0.1, 1.1])
    plt.show()


if __name__ == '__main__':
    gen_result_dir = Path(f"{BASE_DIR}/resources/generation_results")
    # run_examples()

    plot_gnn_scatters(gen_result_dir)

    # plot_lm_scatters(gen_result_dir)
