"""
@Desc:
@Reference:
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

from src.utils.file_utils import save_json, load_json
from src.utils.misc_utils import reverse_dict_key_val
from transformers import BartTokenizer
from tqdm import tqdm

def lines_to_file(lines: list, file_path: Path):
    with file_path.open("w", encoding="utf-8") as fw:
        for line in lines:
            fw.write(f"{line}\n")

def filter_minus_in_list(list_):
    new_ = []
    for num in list_:
        if num >= 0:
            new_.append(num)
    return new_

def file_to_list(file_path: Path):
    res = []
    with file_path.open("r", encoding="utf-8") as fr:
        for line in fr:
            res.append(line.strip())
    return res

def match_entities(post_words, entity_list):
    ents = set()
    for word in post_words:
        if word in entity_list:
            ents.add(word)
    return ents

def examples_process(dataset_dir: Path, output_dir: Path):
    posts = []
    responses = []
    response_ents = []
    triple_nums = []
    post_entities = []
    dataset_path = dataset_dir.joinpath(f"testset.txt")
    entity_list = file_to_list(Path(f"{BASE_DIR}/resources/commonsense_conversation_dataset/entity.txt"))
    for line in tqdm(dataset_path.open("r", encoding="utf-8").readlines(), desc=f"extract examples from {dataset_path.name}"):
        json_sample = json.loads(line)
        posts.append(' '.join(json_sample["post"]))
        responses.append(' '.join(json_sample["response"]))
        post_entities.append(list(match_entities(json_sample["post"], entity_list)))
        response_triples = list(json_sample["response_triples"])
        assert len(response_triples) == len(json_sample["response"])
        ents_ = []
        for tri, res_word in zip(response_triples, json_sample["response"]):
            if tri != -1:
                ents_.append(res_word)
        response_ents.append(ents_)
        all_triples = json_sample["all_triples"]
        triple_nums.append(len(set([t for one in all_triples for t in one])))

    lines_to_file(posts, output_dir.joinpath("testset_posts.txt"))
    lines_to_file(responses, output_dir.joinpath("testset_responses.txt"))
    lines_to_file(response_ents, output_dir.joinpath("response_ents.txt"))
    lines_to_file(triple_nums, output_dir.joinpath("triple_nums.txt"))
    lines_to_file(post_entities, output_dir.joinpath("post_entities.txt"))

    # deal with generated_res.txt
    if output_dir.joinpath("generate_responsetopCCM.txt").exists():
        os.rename(output_dir.joinpath("generate_responsetopCCM.txt"), output_dir.joinpath("CCM_gen.txt"))

    raw_file = output_dir.joinpath("generated_res.txt")
    conceptflow_file = output_dir.joinpath("conceptflow_gen.txt")
    if not conceptflow_file:
        with raw_file.open("r", encoding="utf-8") as fr:
            res_list = []
            for line in fr:
                json_line = json.loads(line.strip())
                res = " ".join(json_line["res_text"])
                res_list.append(res)
            lines_to_file(res_list, conceptflow_file)


def get_statistics(data_dir: Path):
    for dataset_type in ["train", "val", "test"]:
        dataset_name = f"{dataset_type}set.txt"
        data_file = data_dir.joinpath(dataset_name)
        entities = set()
        len_post_entities = []
        len_response_entities = [] # equal to entities in response
        len_subgraghs = []
        triples = []
        pair_count = 0
        vocab = set()
        for line in tqdm(data_file.open("r", encoding="utf-8").readlines(), desc=f"processing {dataset_name}"):
            json_sample = json.loads(line)
            all_entities = json_sample["all_entities"]
            all_triples = json_sample["all_triples"]
            response_triples = json_sample["response_triples"]
            vocab.update(json_sample["post"])
            vocab.update(json_sample["response"])
            # process -----
            pair_count += 1
            len_post_entities.append(len(set([ent for one in all_entities for ent in one])))
            entities.update([ent for one in all_entities for ent in one])
            if -1 in response_triples:
                response_triples.remove(-1)
            len_response_entities.append(len(set(response_triples)))
            entities.update(response_triples)
            len_subgraghs.append(len(all_triples))
            triples.append(set([t for one in all_triples for t in one]))
        # print
        print(f"============ {dataset_name} ==============")
        print(f"pairs: {pair_count}; vocab size: {len(vocab)}")
        ent_set_len = len(entities)
        triple_set_len = len(set([t for one in triples for t in one]))
        print(f"entities: {ent_set_len};  triples: {triple_set_len}")
        print(f"avg. entities in post {sum(len_post_entities)/ pair_count};"
              f"avg. entities in response {sum(len_response_entities)/ pair_count};"
              f"avg. subgraphs in pair {sum(len_subgraghs)/ pair_count}"
              f"avg. triples in pair {sum([len(one) for one in triples])/ pair_count}")





if __name__ == '__main__':
    # # get statistics ----------
    # data_dir = Path(f"{BASE_DIR}/datasets/cadge")
    # get_statistics(data_dir=data_dir)

    # process examples ----------
    data_dir = Path(f"{BASE_DIR}/datasets/cadge")
    output_dir = Path(f"{BASE_DIR}/resources/generation_results")
    examples_process(data_dir, output_dir)
