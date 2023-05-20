"""
@Desc:
@Reference:
"""

import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler

from transformers.models.bart import modeling_bart
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, BartConfig
from transformers import BartTokenizer

from src.utils.gen_utils import ids_to_clean_string, top_p_logits
from src.utils import nlg_eval_utils
from src.utils.charbot import model_utils
from src.models.lightning_base import BaseTransformer
from src.modules.chatbot.dataloader_onehop import (
    CommonGraphDataset
)


logger = logging.getLogger(__name__)


class MyBart(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

        self._custom_init()

        # Whether changing embeddings
        if self.hparams.freeze_embeds:
            model_utils.freeze_embeds(self.model)
        if self.hparams.freeze_encoder:
            model_utils.freeze_params(self.model.get_encoder())
            model_utils.assert_all_frozen(self.model.get_encoder())

        self.step_count = 0
        self.current_val_metrics = {}
        self.metrics_save_path = Path(self.experiment_output_dir) / "metrics.json"
        self.metrics: dict = defaultdict(list)
        self.model_type = self.config.model_type
        self.decoder_start_token_id = self.model.config.decoder_start_token_id  # default to config
        self.already_saved_batch = False  # flag of saving readable batch
        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
        self.val_metric = "loss" if self.hparams.val_metric is None else self.hparams.val_metric
        self.save_readable_batch = True  # for debug
        self.metric_names_update_flag = True

        # predicted
        self.use_top_p = False
        self.top_p = 0.9
        self.store_test_output = True
        self.test_output = None
        self.remain_sp_tokens = self.hparams.remain_sp_tokens
        if self.remain_sp_tokens:
            print("remain special tokens in target and pred text (e.g. [EVENT_s])")

    def _custom_init(self):
        # load pretrained settings from bart
        # config
        self.config: BartConfig = BartConfig.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: BartTokenizer = BartTokenizer.from_pretrained(self.hparams.model_name_or_path)
        # model
        self.model = self._load_model(self.hparams.model_name_or_path, BartForConditionalGeneration, self.config)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.dataset_class = CommonGraphDataset

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def gather_nd(self, x, indices):
        newshape = indices.shape[:-1] + x.shape[indices.shape[-1]:]
        indices = indices.view(-1, indices.shape[-1]).tolist()
        out = torch.cat([x.__getitem__(tuple(i)) for i in indices]).reshape(newshape)
        return out

    def _step(self, batch: dict):
        src_ids = batch['src_ids']
        tgt_ids = batch['tgt_ids']
        encoder_attention = batch['encoder_attention']

        outputs = self(src_ids, attention_mask=encoder_attention, labels=tgt_ids)
        return outputs["loss"]

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss = self._step(batch)
        logs = {"loss": loss.item()}
        # metrics logged can be access by trainer.callback_metrics
        self.log_dict(self.current_val_metrics)
        logs["batch_size"] = batch['src_ids'].shape[0]
        return {"loss": loss, "log": logs}

    def gen_ids_to_clean_text(self, generated_ids: List[int]):
        gen_list = []
        for output in generated_ids:
            gen_list.append(ids_to_clean_string(output, self.tokenizer, remain_sp_tokens=self.remain_sp_tokens))
        return gen_list

    @torch.no_grad()
    def _generative_step(self, batch: dict) -> dict:
        tik = datetime.now()
        src_ids, src_mask = batch['src_ids'], batch["encoder_attention"]
        tgt_ids = batch['tgt_ids']
        encoder_attention = batch['encoder_attention']
        one_hop_graph = batch['graph']
        csk_ids = []
        for _one_hop_graph in one_hop_graph:
            _csk_ids = _one_hop_graph.ndata['h'][:, [0, 3]]
            _csk_ids = set([int(ids) for tri in _csk_ids for ids in tri])
            csk_ids.append(_csk_ids)


        extra_params = {}
        if self.hparams.num_beam_groups > 1:
            extra_params["num_beam_groups"] = self.hparams.num_beam_groups
            extra_params["diversity_penalty"] = self.hparams.diversity_penalty
        if self.eval_beams >= 1:
            extra_params["num_beams"] = self.eval_beams
        if self.hparams.repetition_penalty > 0:
            extra_params["repetition_penalty"] = self.hparams.repetition_penalty

        generated_ids = self.model.generate(
            input_ids=src_ids,
            attention_mask=encoder_attention,
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

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        generative_metrics = {
            name: np.array([x[name] for x in outputs]).mean() for name in self.metric_names
        }
        metric_val = (
            torch.tensor(generative_metrics[self.val_metric])
        )
        val_metrics = {f"{prefix}_{k}": x for k, x in generative_metrics.items()}
        val_metrics["step_count"] = float(self.step_count)
        self.current_val_metrics = val_metrics
        self.metrics[prefix].append(val_metrics)  # callback writes this to self.metrics_save_path.
        print(f"Evaluation result: {val_metrics}")
        preds = model_utils.flatten_list([x["preds"] for x in outputs])
        tgts = model_utils.flatten_list([x["targets"] for x in outputs])
        self.log_dict(self.current_val_metrics)
        return {
            "log": val_metrics,
            "preds": preds,
            "tgts": tgts,
            f"{prefix}_loss": generative_metrics["loss"],
            f"{prefix}_{self.val_metric}": metric_val,
        }

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        test_output = self.validation_epoch_end(outputs, prefix="test")
        if self.store_test_output:
            self.test_output = test_output
        return test_output

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

    def get_dataloader(self, data_name: str,  batch_size: int, shuffle: bool = False):
        dataset = self.get_dataset(data_name, batch_size)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        train_shuffle = True if self.hparams.overfit_batches == 0.0 else False
        if not train_shuffle:
            print(f"train_shuffle: {train_shuffle} overfit_batches: {self.hparams.overfit_batches}")
        return self.get_dataloader("train", batch_size=self.hparams.train_batch_size,
                                   shuffle=train_shuffle)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size,
                                           shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size,
                                   shuffle=False)