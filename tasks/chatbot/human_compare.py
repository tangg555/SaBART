"""
@Desc:
@Reference:
@Notes:
"""
import random
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
from src.utils.nlg_eval_utils import calculate_bleu
from analyze import file_to_list
from collections import Counter
random.seed(42)
np.random.seed(42)

def human_eval(gen_result_dir: Path):
    testset_posts = file_to_list(gen_result_dir.joinpath("testset_posts.txt"))
    testset_responses = file_to_list(gen_result_dir.joinpath("testset_responses.txt"))
    chatbot_onehop_gen = file_to_list(gen_result_dir.joinpath("chatbot_onehop_gen.txt"))
    chatbot_onehop_no_kgagg_gen = file_to_list(gen_result_dir.joinpath("chatbot_onehop_no_agg_gen.txt"))
    chatbot_onehop_no_child_gen = file_to_list(gen_result_dir.joinpath("chatbot_onehop_no_child_gen.txt"))
    conceptflow_gen = file_to_list(gen_result_dir.joinpath("conceptflow_gen.txt"))
    mybart_gen = file_to_list(gen_result_dir.joinpath("mybart_gen.txt"))


    score_pair = [(idx, calculate_bleu(ref_lines=[testset_responses[idx]],
                                       gen_lines=[chatbot_onehop_gen[idx]])['bleu-2'])
                  for idx in range(len(testset_responses))]
    score_pair.sort(key=lambda x: x[1], reverse=True)
    scorer = Counter()
    for pair in score_pair[:100]:
        idx, score = pair
        print(f"========={idx} score: {score}==============")
        print(f"post: {testset_posts[idx]}")
        print(f"testset_responses: {testset_responses[idx]}")
        print(f"=======================")
        print(f"1.chatbot_onehop_gen: {chatbot_onehop_gen[idx]}")
        print(f"2.chatbot_onehop_no_kgagg_gen: {chatbot_onehop_no_kgagg_gen[idx]}")
        print(f"3.chatbot_onehop_no_child_gen: {chatbot_onehop_no_child_gen[idx]}")
        print(f"4.conceptflow_gen: {conceptflow_gen[idx]}")
        print(f"5.mybart_gen: {mybart_gen[idx]}")
        scorer[int(input("who win?"))] += 1
    print(scorer)

def show_post_entity(text, post_entities):
    ents = []
    for word in text.split():
        word = word.strip()
        if word in post_entities:
            ents.append(word)
    return ents

def show(gen_result_dir: Path):
    testset_posts = file_to_list(gen_result_dir.joinpath("testset_posts.txt"))
    testset_responses = file_to_list(gen_result_dir.joinpath("testset_responses.txt"))
    chatbot_onehop_gen = file_to_list(gen_result_dir.joinpath("chatbot_onehop_gen.txt"))
    chatbot_onehop_no_kgagg_gen = file_to_list(gen_result_dir.joinpath("chatbot_onehop_no_agg_gen.txt"))
    chatbot_onehop_no_child_gen = file_to_list(gen_result_dir.joinpath("chatbot_onehop_no_child_gen.txt"))
    conceptflow_gen = file_to_list(gen_result_dir.joinpath("conceptflow_gen.txt"))
    mybart_gen = file_to_list(gen_result_dir.joinpath("mybart_gen.txt"))

    post_entities = []
    with gen_result_dir.joinpath("post_entities.txt").open("r", encoding="utf-8") as fr:
        for line in fr:
            post_entities.append(eval(line.strip()))

    while True:
        idx = int(input("the index"))

        print(f"=======================")
        print(f"post: {testset_posts[idx]}; ents: {show_post_entity(testset_posts[idx], post_entities[idx])}")
        print(f"testset_responses: {testset_responses[idx]}")
        print(f"=======================")
        print(f"1.chatbot_onehop_gen: {chatbot_onehop_gen[idx]}; ents: {show_post_entity(chatbot_onehop_gen[idx], post_entities[idx])}")
        print(f"2.chatbot_onehop_no_kgagg_gen: {chatbot_onehop_no_kgagg_gen[idx]}; ents: {show_post_entity(chatbot_onehop_no_kgagg_gen[idx], post_entities[idx])}")
        print(f"3.chatbot_onehop_no_child_gen: {chatbot_onehop_no_child_gen[idx]}; ents: {show_post_entity(chatbot_onehop_no_child_gen[idx], post_entities[idx])}")
        print(f"4.conceptflow_gen: {conceptflow_gen[idx]}; ents: {show_post_entity(chatbot_onehop_gen[idx], conceptflow_gen[idx])}")


if __name__ == '__main__':
    gen_result_dir = Path(f"{BASE_DIR}/resources/generation_results")
    show(gen_result_dir)