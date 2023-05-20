"""
@ref:
ast.literal_eval https://blog.csdn.net/Jerry_1126/article/details/68831254
DGL nodes and edges https://docs.dgl.ai/en/0.8.x/guide_cn/graph-graphs-nodes-edges.html
max(set(l), key=l.count) https://www.reddit.com/r/learnpython/comments/5wsjra/why_does_maxsetl_keylcount_generate_the_mode_of_a/
"""

import numpy as np
import torch
import json
import dgl
from pathlib import Path

from src.utils.file_utils import save_json, load_json
from src.utils.misc_utils import reverse_dict_key_val
from transformers import BartTokenizer

class CommonGraphDataset(torch.utils.data.Dataset):
    NO_RET_TOK = "[NO_RET]"
    TRIPLE_BOS = "[T_BOS]"

    def __init__(self, data_dir, hparams, batch_size, tokenizor, embedding_size=768, data_name='train'):
        assert data_name in ['train', 'test', 'val'], "Data name should be among ['train', 'test', 'val']."
        print(f"====== initializing the dataset =======")
        self.hparams = hparams
        self.data_dir = Path(data_dir)
        self.data_name = data_name
        self.embedding_size = embedding_size
        self.word2id = load_json(self.data_dir.joinpath("word2id.txt"))
        self.ent2id = self.word2id
        self.rel2word = load_json(self.data_dir.joinpath("rel2word.txt"))
        self.batch_size = batch_size
        self.tokenizer = tokenizor
        self.triple2id = load_json(self.data_dir.joinpath("triple2id.txt"))
        self.id2word = self.json_key_str_to_id(load_json(self.data_dir.joinpath("id2word.txt")))
        self.id2triple = self.json_key_str_to_id(load_json(self.data_dir.joinpath("id2triple.txt")))
        self.tokenizer.add_special_tokens({'additional_special_tokens':
                                               [self.NO_RET_TOK, self.TRIPLE_BOS],
                                           })
        self.NO_RET_TOK_ID = self.tokenizer._convert_token_to_id(self.NO_RET_TOK)
        self.TRIPLE_BOS_ID = self.tokenizer._convert_token_to_id(self.TRIPLE_BOS)
        self.PAD_ID = self.tokenizer.pad_token_id
        self.NO_RET_TRIPLE = [self.NO_RET_TOK_ID, self.NO_RET_TOK_ID, self.NO_RET_TOK_ID, self.NO_RET_TOK_ID]
        self.BOS_TRIPLE = [self.TRIPLE_BOS_ID, self.TRIPLE_BOS_ID, self.TRIPLE_BOS_ID, self.TRIPLE_BOS_ID,]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_src_len = self.hparams.max_src_len
        self.max_tgt_len = self.hparams.max_tgt_len
        # max_limit is only used when debugging
        self.convdata = self.load_conv_data(self.data_dir.joinpath(f"{data_name}set.txt"), max_limit=1000000)
        print(f"词向量大小: {len(self.word2id)}")
        print(f"There are {len(self.convdata)} conversations in total.")
        print(f"====== dataset initialized =======")

    def json_key_str_to_id(self, dict_obj: dict):
        new_dict = {}
        for key, val in dict_obj.items():
            new_dict[int(key)] = val
        return new_dict

    def load_conv_data(self, data_path: Path, max_limit=None):
        data_list = []
        with data_path.open("r", encoding="utf-8") as fr:
            count = 0
            for line in fr:
                count += 1
                one_data = json.loads(line.strip())
                data_list.append(one_data)
                if max_limit and count >= max_limit:
                    break
        print(f"data from {data_path} size: {len(data_list)}; max_limit is set to {max_limit}")
        return data_list

    def word2id_helper(self, word: str):
        if f"Ġ{word}" in self.word2id:
            return self.word2id[f"Ġ{word}"]
        elif word in self.word2id:
            return self.word2id[word]
        else:
            return self.tokenizer.unk_token_id

    def process_triple(self, all_triples, target_triples):
        """
        Returns:
            src_triples: for each data sample [[head_it, rel_id, rel_id, tail_id], ...]
            main_entities: the entity appears most times for each data sample
            target_ents: for each data sample [head_ids] + [tail_ids]
        """
        src_triples = []
        main_entities = []
        for triple_list in all_triples:
            processed_triple = []
            for t in triple_list:
                triple = self.id2triple[t].split(', ')
                head = triple[0]
                rel = triple[1]
                tail = triple[2]
                rel = self.rel2word[rel].split()
                processed_triple.append(
                    [self.word2id_helper(head),
                     self.word2id_helper(rel[0]),
                     self.word2id_helper(rel[1]),
                     self.word2id_helper(tail)])
            src_triples.append(processed_triple)
            entities = [p for triple in processed_triple for p in triple]
            if len(entities) > 0:
                entity = max(set(entities), key=entities.count)
            else:
                entity = None
            main_entities.append(entity)
        target_head_ents = []
        target_tail_ents = []
        for t in target_triples:
            triple = self.id2triple[t].split(', ')
            head = triple[0]
            tail = triple[2]
            if head in self.word2id and tail in self.word2id:
                target_head_ents.append(self.word2id[head])
                target_tail_ents.append(self.word2id[tail])
        target_ents = target_head_ents + target_tail_ents
        return src_triples, main_entities, target_ents

    def make_graph(self, all_triples, main_entities, target_ents):
        """
        embedding: list of triples

        """
        word2node_id = {}
        edges = []
        target_node = []
        embedding = []
        ##add a root node with 0 embeddings
        embedding.append(self.BOS_TRIPLE)
        word2node_id['root'] = len(word2node_id)
        target_node.append(1)
        for data_index, triples in enumerate(all_triples):
            main_ent = main_entities[data_index]
            if main_ent is self.NO_RET_TOK_ID:
                continue
            if main_ent not in word2node_id:
                word2node_id[main_ent] = len(word2node_id)
                e = self.BOS_TRIPLE
                embedding.append(e)
                if main_ent in target_ents:
                    target_node.append(1)
                else:
                    target_node.append(0)
            edges.append([0, word2node_id[main_ent]])

            for t_index, t in enumerate(triples):
                # define the attribute of the tail nodes and connect them to the head node
                if str(t) not in word2node_id:
                    word2node_id[str(t)] = len(word2node_id)
                    # e = self.ent_embedding(torch.tensor(t)).flatten()
                    embedding.append(t)
                    if t[0] in target_ents and t[-1] in target_ents:
                        target_node.append(1)
                    else:
                        target_node.append(0)
                edges.append([word2node_id[main_ent], word2node_id[str(t)]])
        # graph(u, v)
        if len(edges) > 0:
            edges = torch.tensor(edges)
            g = dgl.graph((edges[:, 1], edges[:, 0]), num_nodes=len(word2node_id))
            g.ndata['h'] = torch.tensor(embedding).detach()
            g.ndata['_ID'] = torch.Tensor(np.array(range(len(word2node_id)))).long()
            tgt_tensor = torch.tensor(target_node, dtype=torch.float32)
        else:
            edges = [[0, 1]]
            edges = torch.tensor(edges)
            g = dgl.graph((edges[:, 1], edges[:, 0]), num_nodes=2)
            g.ndata['h'] = self.NO_RET_TOK_ID * torch.ones((2, 4)).long()
            g.ndata['_ID'] = torch.Tensor(np.array(range(2))).long()
            tgt_tensor = torch.tensor([0, 0], dtype=torch.float32)
        g.ndata['tgt'] = tgt_tensor
        g.ndata['loss'] = torch.zeros_like(tgt_tensor)
        g.ndata['tgt_num'] = torch.zeros_like(tgt_tensor)

        return g

    def pad_batch_data(self, insts, pad_id):
        """Pad the instances to the max sequence length in batch. """
        max_len = max(map(len, insts))
        inst_data = np.array([list(inst) + [pad_id] * (max_len - len(inst)) for inst in insts])
        return torch.Tensor(inst_data).reshape([-1, max_len])

    def __len__(self):
        return len(self.convdata)

    def __getitem__(self, i):
        data = self.convdata[i]

        src_text = " ".join(data['post'])
        tgt_text = " ".join(data['response'])

        src_triples, main_entity, target_ent = self.process_triple(data['all_triples'], data['match_triples'])
        graph = self.make_graph(src_triples, main_entity, target_ent)

        return {'src_text': src_text, 'tgt_text': tgt_text, 'graph': graph}

    def collate_fn(self, batch):
        x_encoding = self.tokenizer(
            [x["src_text"] for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.hparams.max_src_len,
            return_tensors="pt",
        ).data

        label_encoding = self.tokenizer(
            [x["tgt_text"] for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.hparams.max_tgt_len,
            return_tensors="pt",
        ).data

        graph = [s['graph'] for s in batch]  # (bsz,)

        batched_data = {
            'src_ids': x_encoding["input_ids"],
            'tgt_ids': label_encoding["input_ids"],
            'encoder_attention': x_encoding["attention_mask"],
            'decoder_attention': label_encoding["attention_mask"],
            'graph': graph
        }
        return batched_data

