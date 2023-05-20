import sys

import torch
import torch.nn as nn
import dgl
import numpy as np
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.file_utils import ModelOutput
from transformers import BartTokenizer
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, Seq2SeqModelOutput, Seq2SeqLMOutput, \
    shift_tokens_right
from src.modules.chatbot.graph_attention import GraphAttnLayer, GlobalGraphAttnLayer


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class OneHopGNNBartForCG(BartForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')

        self.graph_drop = 0 if not hasattr(self.config, "graph_drop") else self.config.graph_drop
        self.leaky_negative_slope = 0.98 if not hasattr(self.config, "leaky_negative_slope") \
            else self.config.leaky_negative_slope
        # one-hop
        self.q_layer = nn.Linear(self.config.hidden_size + config.d_model,
                                 self.config.hidden_size + config.d_model,  bias=True)
        self.graph_fc = nn.Linear(config.d_model * 4, config.d_model * 4, bias=False)
        self.graph_model = GlobalGraphAttnLayer(config.d_model * 4, config.d_model * 4, config.hidden_size,
                                          self.graph_drop, self.leaky_negative_slope)
        self.init_weights()
        self.tie_weights()
        self.config.is_encoder_decoder = False


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

            # --------------
            # g.ndata['q']: (nodes, embed)
            graph_batch = dgl.batch(one_hop_graph)

            # g.ndata['h']: (nodes, 4*embed)
            graph_batch.ndata['h'] = self.get_input_embeddings()(graph_batch.ndata['h']).view(-1, self.config.hidden_size * 4)
            # fully connected -> g.ndata['h']: (nodes, 4*embed)
            graph_batch.ndata['h'] = self.graph_fc(graph_batch.ndata['h'])

            graphs = dgl.unbatch(graph_batch)
            child_node_knowledge = torch.stack([torch.mean(g.ndata['h'], dim=0).view((4, -1)) for g in graphs])
            # --------------

            q_features = self.q_layer(torch.cat((encoded_outputs[0][:,0],
                                          torch.max(child_node_knowledge, dim=1)[0]), dim=1))

            for gindex, g in enumerate(graphs):
                g.ndata['q'] = q_features[gindex].expand((g.num_nodes(),
                                                          self.config.hidden_size + self.config.d_model))

            graph_batch = dgl.batch(graphs)

            graph_batch = self.graph_model(graph_batch)

            graphs = dgl.unbatch(graph_batch)
            one_hop_knowledge_distill = torch.stack([g.ndata['h'][0].view((4, -1)) for g in graphs])

            # --------------
            encoder_last_hidden_state = torch.cat((one_hop_knowledge_distill, child_node_knowledge, encoded_outputs[0]), dim=1)
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

            # # one hop loss - bad performance
            # es_loss = sum([g.ndata['loss'][0] / (g.ndata['tgt_num'][0] + 1) for g in graphs])
            # es_loss = es_loss / len(graphs)
            # loss += es_loss

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

    def encoder_outputs(self,one_hop_graph, input_ids, encoder_attention=None):
        encoded_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=encoder_attention
        )
        cls = self.q_layer(encoded_outputs[0][:, 0])

        for gindex, g in enumerate(one_hop_graph):
            g.ndata['q'] = cls[gindex].expand((g.num_nodes(), self.config.hidden_size))

        graph_batch = dgl.batch(one_hop_graph)

        graph_batch = self.graph_model(graph_batch)
        graphs = dgl.unbatch(graph_batch)
        one_hop_knowledge_distill = torch.stack([g.ndata['h'][0].view((4, -1)) for g in graphs])

        hidden_states = torch.cat((one_hop_knowledge_distill, encoded_outputs[0]), dim=1)
        encoded_outputs.last_hidden_state = hidden_states

        return encoded_outputs

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            src_ids=None,
            past=None,
            one_hop_graph=None,
            encoder_last_hidden_state=None,
            encoder_attention=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used


        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "src_ids":src_ids,
            "one_hop_graph":one_hop_graph,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "encoder_last_hidden_state": encoder_last_hidden_state,
            "encoder_attention": encoder_attention,
        }


    def _update_model_kwargs_for_generation(self,
        outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        if "encoder_last_hidden_state" in outputs:
            model_kwargs["encoder_last_hidden_state"] = outputs.encoder_last_hidden_state
        if "encoder_attentions" in outputs:
            model_kwargs["encoder_attentions"] = outputs.encoder_attentions

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return model_kwargs
