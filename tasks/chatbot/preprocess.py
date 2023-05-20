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

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from transformers import BartTokenizer
from tqdm import tqdm

from src.configuration.chatbot.config_args import parse_args_for_config
from src.utils.file_utils import load_json, save_json
from src.utils.misc_utils import reverse_dict_key_val
from src.utils.huggingface_helper import download_repository
from src.utils.file_utils import copy_dir, rm_dir

class CadgeDatasetPreprocessor(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.resource_dir = Path(f"{BASE_DIR}/resources")
        self.data_src_dir = Path(f"{BASE_DIR}/resources/commonsense_conversation_dataset")
        self.data_tgt_dir = Path(f"{BASE_DIR}/datasets/cadge")
        self.model_name_or_path = self.hparams.model_name_or_path
        self.local_model_dir = self.get_local_model()

    def get_local_model(self):
        if not os.path.exists(self.model_name_or_path):
            local_model_dir = self.resource_dir.joinpath(f"external_models/{self.model_name_or_path}")
            if os.path.exists(local_model_dir):
                rm_dir(local_model_dir)
            print(f"==== downloading {self.model_name_or_path} from Huggingface to local dir {local_model_dir} =====")
            downloaded_local_folder = download_repository(repository_id=self.model_name_or_path)
            # clean the dest dir, otherwise it will have an error
            copy_dir(source_path=downloaded_local_folder, target_path=local_model_dir)
            rm_dir(downloaded_local_folder)
        else:
            local_model_dir = self.model_name_or_path
        print(f"local model dir: {local_model_dir}")
        return local_model_dir



    def load_data_plaintext_file(self, file_path: Path):
        with file_path.open("r", encoding="utf-8") as fr:
            list_obj = [line.strip() for line in fr.readlines()]
        return list_obj

    def add_ent_words_to_tokenizer(self, entity_list, tokenizer):
        ent2id = {}
        no_ent_count = 0
        for ent in tqdm(entity_list, desc="add_ent_words_to_tokenizer"):
            if (ent not in tokenizer.get_vocab()) and (f"Ġ{ent}" not in tokenizer.get_vocab()):
                no_ent_count += 1
                tokenizer.add_tokens([f"Ġ{ent}"])
                ent2id[ent] = tokenizer.get_vocab()[f"Ġ{ent}"]

            # add tokens ---
            if f"Ġ{ent}" in tokenizer.get_vocab():
                    ent2id[ent] = tokenizer.get_vocab()[f"Ġ{ent}"]
            elif f"{ent}" in tokenizer.get_vocab():
                    ent2id[ent] = tokenizer.get_vocab()[ent]
            else:
                raise ValueError()

        print(f"ent not in vocab: {no_ent_count}/{len(entity_list)} ratio: {round(no_ent_count / len(entity_list), 2)}")

    def add_rel_words_to_tokenizer(self, relation_words, tokenizer):
        no_rel_word_count = 0
        for rel_word in tqdm(relation_words, desc="add_rel_words_to_tokenizer"):
            if (rel_word not in tokenizer.get_vocab()) and (f"Ġ{rel_word}" in tokenizer.get_vocab()):
                no_rel_word_count += 1
            # add tokens ---
            if rel_word not in tokenizer.get_vocab():
                tokenizer.add_tokens([rel_word])
            if f"Ġ{rel_word}" not in tokenizer.get_vocab():
                tokenizer.add_tokens([f"Ġ{rel_word}"])
        print(f"rel_word not in vocab: {no_rel_word_count}/{len(relation_words)} "
              f"ratio: {round(no_rel_word_count / len(relation_words), 2)}")

    def preprocess_cadge_data(self):
        """
        ------------ single hop -------------
        data dir should have:
        word2id: ['fawn': 0, ...]
        id2word: [0: 'fawn', ...]
        rel2word: ['AtLocation': "at location", ...]
        csk_dict: [fawn: ['deer', RelatedTo, fawn', ...], ...]
        """

        self.data_tgt_dir.mkdir(exist_ok=True, parents=True)
        tokenizer = BartTokenizer.from_pretrained(self.local_model_dir)
        resource_file = self.data_src_dir.joinpath("resource.txt")
        resource_dict = json.load(resource_file.open("rb"))

        print(f"original vocab size of Bart: {len(tokenizer.get_vocab())}")
        entity_list = self.load_data_plaintext_file(self.data_src_dir.joinpath("entity.txt"))
        relation_words = []
        rel2words: dict = load_json(Path(f"{BASE_DIR}/resources/rel2words.txt"))
        for list_obj in rel2words.values():
            relation_words.extend(list_obj)

        # add new tokens of entities to vocab, and get ent2id -----------
        self.add_ent_words_to_tokenizer(entity_list, tokenizer)

        # add new tokens of relations to vocab -----------tokenizer
        self.add_rel_words_to_tokenizer(relation_words, tokenizer)

        word2id = tokenizer.get_vocab()
        id2word = reverse_dict_key_val(word2id)

        triple2id = resource_dict["dict_csk_triples"]
        id2triple = reverse_dict_key_val(triple2id)
        save_json(triple2id, self.data_tgt_dir.joinpath("triple2id.txt"))
        save_json(id2triple, self.data_tgt_dir.joinpath("id2triple.txt"))
        save_json(rel2words, self.data_tgt_dir.joinpath("rel2word.txt"))
        save_json(word2id, self.data_tgt_dir.joinpath("word2id.txt"))
        save_json(id2word, self.data_tgt_dir.joinpath("id2word.txt"))
        print(f"We have added new tokens: {len(tokenizer.get_added_vocab())}")
        tokenizer.save_pretrained(self.local_model_dir)
        print(f"new tokenizer has been saved to {hparams.model_name_or_path}")

        # copy data files

        shutil.copyfile(self.data_src_dir.joinpath("trainset.txt"), self.data_tgt_dir.joinpath("trainset.txt"))
        shutil.copyfile(self.data_src_dir.joinpath("validset.txt"), self.data_tgt_dir.joinpath("testset.txt"))
        shutil.copyfile(self.data_src_dir.joinpath("testset.txt"), self.data_tgt_dir.joinpath("valset.txt"))

if __name__ == '__main__':
    hparams = parse_args_for_config()
    # 最好在命令行传入这个值
    # hparams.model_name_or_path="facebook/bart-base"
    preprocessor = CadgeDatasetPreprocessor(hparams)
    preprocessor.preprocess_cadge_data()

