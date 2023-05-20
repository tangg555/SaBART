"""
@Desc:
@Reference:
"""

import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import dgl

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler

from transformers.models.bart import modeling_bart
from transformers.models.bart.modeling_bart import BartConfig
from transformers import BartTokenizer

from src.utils.gen_utils import ids_to_clean_string, top_p_logits
from src.utils import nlg_eval_utils
from src.modules.chatbot.chatbot_onehop import OneHopGNNBartForCG
from src.utils.charbot import model_utils
from src.models.lightning_base import BaseTransformer
from src.models.chabot.bart import MyBart
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, Seq2SeqModelOutput, Seq2SeqLMOutput, \
    shift_tokens_right
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from src.modules.chatbot.graph_attention import GraphAttnLayer
import torch.nn as nn
from src.modules.chatbot.dataloader_onehop import (
    CommonGraphDataset
)

logger = logging.getLogger(__name__)

class OneHopGNNBartForCGNoChild(OneHopGNNBartForCG):
    def __init__(self, config):
        super().__init__(config)
        self.q_layer = nn.Linear(self.config.hidden_size,
                                 self.config.hidden_size,  bias=True)
        self.graph_model = GraphAttnLayer(config.d_model * 4, config.d_model * 4, config.hidden_size,
                                          self.graph_drop, self.leaky_negative_slope)
        self.init_weights()
        self.tie_weights()

    def forward(self, one_hop_graph,
                src_ids,
                tgt_ids=None,
                encoder_attention=None, decoder_attention=None,
                encoder_last_hidden_state = None,
                decoder_input_ids=None, past_key_values=None,**kwargs):
        if encoder_last_hidden_state is None:
            if tgt_ids is not None:
                if decoder_input_ids is None:
                    decoder_input_ids = shift_tokens_right(
                        tgt_ids, self.config.pad_token_id, self.config.decoder_start_token_id
                    )
            encoded_outputs = self.model.encoder(
                input_ids = src_ids,
                attention_mask = encoder_attention
            )
            q_features = self.q_layer(encoded_outputs[0][:, 0])

            for gindex, g in enumerate(one_hop_graph):
                g.ndata['q'] = q_features[gindex].expand((g.num_nodes(), self.config.hidden_size))

            # --------------
            # g.ndata['q']: (nodes, embed)
            graph_batch = dgl.batch(one_hop_graph)

            # g.ndata['h']: (nodes, 4*embed)
            graph_batch.ndata['h'] = self.get_input_embeddings()(graph_batch.ndata['h']).view(-1, self.config.hidden_size * 4)
            # fully connected -> g.ndata['h']: (nodes, 4*embed)
            graph_batch.ndata['h'] = self.graph_fc(graph_batch.ndata['h'])

            graph_batch = self.graph_model(graph_batch)

            graphs = dgl.unbatch(graph_batch)
            one_hop_knowledge_distill = torch.stack([g.ndata['h'][0].view((4, -1)) for g in graphs])

            # --------------
            encoder_last_hidden_state =  torch.cat((one_hop_knowledge_distill, encoded_outputs[0]), dim=1)
            if encoder_last_hidden_state.shape[0] != decoder_input_ids.shape[0]:
                expanded_return_idx = (
                    torch.arange(encoder_last_hidden_state.shape[0]).view(-1, 1).repeat(1, int(decoder_input_ids.shape[0] /
                        encoder_last_hidden_state.shape[0])).view(-1).to(encoder_last_hidden_state.device)
                )
                encoder_last_hidden_state = encoder_last_hidden_state.index_select(0, expanded_return_idx)

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_last_hidden_state,
            attention_mask=decoder_attention,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True, output_hidden_states=True, return_dict=True)

        lm_logits = self.lm_head(decoder_outputs[0]) + self.final_logits_bias

        loss = None
        if tgt_ids is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), tgt_ids.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            encoder_last_hidden_state=encoder_last_hidden_state,
            encoder_attentions=encoder_attention,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions
        )

class OneHopGNNBartNoChild(MyBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        # load pretrained settings from bart
        # config
        self.config: BartConfig = BartConfig.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: BartTokenizer = BartTokenizer.from_pretrained(self.hparams.model_name_or_path)
        # model
        self.model = self._load_model(self.hparams.model_name_or_path, OneHopGNNBartForCGNoChild, self.config)
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.dataset_class = CommonGraphDataset
        self.training_type = self.hparams.training_type

    def forward(self, **kwargs):
        return self.model( **kwargs)

    def _step(self, batch: dict):
        src_ids = batch['src_ids']
        tgt_ids = batch['tgt_ids']
        encoder_attention = batch['encoder_attention']
        decoder_attention = batch['decoder_attention']
        one_hop_graph = batch['graph']

        outputs = self(one_hop_graph=one_hop_graph,
                       src_ids=src_ids,
                       tgt_ids=tgt_ids,
                       encoder_attention=encoder_attention,
                       decoder_attention=decoder_attention)

        loss = outputs["loss"]
        return loss

    @torch.no_grad()
    def _generative_step(self, batch: dict) -> dict:
        tik = datetime.now()
        '''src_ids, src_mask = batch['input_ids'], batch["attention_mask"]
        item_nums = batch['item_nums']
        encoder_outputs = self.model.get_encoder()(input_ids=src_ids, attention_mask=src_mask, item_nums=item_nums)'''
        src_ids = batch['src_ids']
        tgt_ids = batch['tgt_ids']
        encoder_attention = batch['encoder_attention']
        decoder_attention = batch['decoder_attention']
        one_hop_graph = batch['graph']
        csk_ids = []
        for _one_hop_graph in one_hop_graph:
            _csk_ids = _one_hop_graph.ndata['h'][:, [0, 3]]
            _csk_ids = set([int(ids) for tri in _csk_ids for ids in tri])
            csk_ids.append(_csk_ids)

        # transformers.generation_utils
        extra_params = {}
        if self.hparams.num_beam_groups > 1:
            extra_params["num_beam_groups"] = self.hparams.num_beam_groups
            extra_params["diversity_penalty"] = self.hparams.diversity_penalty
        if self.eval_beams >= 1:
            extra_params["num_beams"] = self.eval_beams
        if self.hparams.repetition_penalty > 0:
            extra_params["repetition_penalty"] = self.hparams.repetition_penalty

        generated_ids = self.model.generate(
            input_ids=torch.tensor([self.tokenizer.bos_token_id for _
                                    in range(len(tgt_ids))])[:, None].to(src_ids.device),
            src_ids=src_ids,
            encoder_attention=encoder_attention,
            one_hop_graph = one_hop_graph,
            decoder_start_token_id=self.decoder_start_token_id,
            max_length=self.hparams.gen_max_len,
            min_length=self.hparams.gen_min_len,
            top_p=self.top_p if self.use_top_p else None,
            **extra_params
        )

        tok = datetime.now()
        batch_gen_time = tok - tik
        preds: List[str] = self.gen_ids_to_clean_text(generated_ids)
        targets: List[str] = self.gen_ids_to_clean_text(tgt_ids)
        loss = self._step(batch)

        base_metrics = {"loss": loss.item()}

        rouge_metrics: Dict = nlg_eval_utils.calculate_rouge(pred_lines=preds, tgt_lines=targets)
        base_metrics.update(**rouge_metrics)
        bleu_metrics: Dict = nlg_eval_utils.calculate_bleu(ref_lines=[self.tokenizer.tokenize(l) for l in targets],
                                                           gen_lines=[self.tokenizer.tokenize(l) for l in preds])
        base_metrics.update(**bleu_metrics)

        entity_score_metrics: Dict = nlg_eval_utils.compute_ent_score(src_ids, generated_ids, csk_ids)

        base_metrics.update(**entity_score_metrics)
        summ_len = np.mean(list(map(len, generated_ids)))

        # update metric_names
        self.update_metric_names(base_metrics, update_flag=self.metric_names_update_flag)
        self.metric_names_update_flag = False
        base_metrics.update(batch_gen_time=batch_gen_time, gen_len=summ_len,
                            preds=preds, targets=targets)
        return base_metrics

    def training_step(self, batch, batch_idx) -> Dict:
        outputs = self._step(batch)
        loss = outputs
        logs = {"loss": loss.item()}
        # metrics logged can be access by trainer.callback_metrics
        self.log_dict(self.current_val_metrics)
        logs["batch_size"] = batch['src_ids'].shape[0]
        return {"loss": loss, "log": logs}

    def get_dataset(self, data_name: str, batch_size: int) -> CommonGraphDataset:
        dataset = self.dataset_class(
            data_dir=self.hparams.data_dir,
            hparams=self.hparams,
            batch_size=batch_size,
            tokenizor=self.tokenizer,
            embedding_size=self.config.hidden_size,
            data_name=data_name,
        )
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        return dataset

