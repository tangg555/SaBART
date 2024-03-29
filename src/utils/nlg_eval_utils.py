"""
@Desc:
@Reference:
@Notes:
"""

import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Tuple, Union
import numpy as np

from rouge_score import rouge_scorer, scoring
import nltk

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

def line_normalize(line: str):
    line = " ".join(line.strip().split())
    return line

def calculate_bleu(ref_lines, gen_lines, metrics: dict = None):
    if metrics is None:
        metrics = {}
    for bleu_i in range(1, 5):
        weights = tuple([1. / bleu_i for _ in range(bleu_i)])
        metrics[f"bleu-{bleu_i}"] = round(nltk.translate.bleu_score.corpus_bleu(
            list_of_references=[[ref] for ref in ref_lines],
            hypotheses=gen_lines,
            weights=weights), 4)
    return metrics

def extract_rouge_mid_statistics(dct):
    new_dict = {}
    for k1, v1 in dct.items():
        mid = v1.mid
        new_dict[k1] = {stat: round(getattr(mid, stat), 4) for stat in ["precision", "recall", "fmeasure"]}
    return new_dict


def compute_ent_score(src_ids,pred_ids,csk_ids):
    if len(src_ids)==0:
        return {'ent_score':0,'ent_score_new':0}
    ent_count = []
    not_in_context_count = []
    for _src_ids,_pred_ids,_csk_ids in zip(src_ids,pred_ids,csk_ids):
        _ent_count=0
        _not_in_context_count=0
        for w in _pred_ids:
            if int(w) in _csk_ids:
                _ent_count += 1
                if int(w) not in _src_ids:
                    _not_in_context_count += 1
        ent_count.append(_ent_count)
        not_in_context_count.append(_not_in_context_count)
    return {'ent_score':sum(ent_count)/len(ent_count),'ent_score_new':sum(not_in_context_count)/len(not_in_context_count)}



def calculate_rouge(
    pred_lines: List[str],
    tgt_lines: List[str],
    use_stemmer=True,
    rouge_keys=ROUGE_KEYS,
    return_precision_and_recall=False,
    bootstrap_aggregation=True,
    newline_sep=True,
) -> Dict:
    """Calculate rouge using rouge_scorer package.

    Args:
        pred_lines: list of summaries generated by model
        tgt_lines: list of groundtruth summaries (e.g. contents of val.target)
        use_stemmer:  Bool indicating whether Porter stemmer should be used to
        strip word suffixes to improve matching.
        rouge_keys:  which metrics to compute, defaults to rouge1, rouge2, rougeL, rougeLsum
        return_precision_and_recall: (False) whether to also return precision and recall.
        bootstrap_aggregation: whether to do the typical bootstrap resampling of scores. Defaults to True, if False
            this function returns a collections.defaultdict[metric: list of values for each observation for each subscore]``
        newline_sep:(default=True) whether to add newline between sentences. This is essential for calculation rougeL
        on multi sentence summaries (CNN/DM dataset).

    Returns:
         Dict[score: value] if aggregate else defaultdict(list) keyed by rouge_keys

    """
    scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()
    for pred, tgt in zip(tgt_lines, pred_lines):
        # rougeLsum expects "\n" separated sentences within a summary
        if newline_sep:
            pred = pred + "\n"
            tgt = tgt + "\n"
        scores = scorer.score(pred, tgt)
        aggregator.add_scores(scores)

    if bootstrap_aggregation:
        result = aggregator.aggregate()
        if return_precision_and_recall:
            return extract_rouge_mid_statistics(result)  # here we return dict
        else:
            return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    else:
        return aggregator._scores  # here we return defaultdict(list)


def repetition_distinct_metric(gen_lines, metrics: dict = None, repetition_times=2):
    if metrics is None:
        metrics = {}

    for gram_n in range(1, 5):
        repetition_count = 0
        all_ngram = defaultdict(int)
        all_ngram_num = 0
        for gen_idx, line in enumerate(gen_lines):
            n_grams = ["_".join(gram) for gram in nltk.ngrams(line, n=gram_n)]
            all_ngram_num += len(n_grams)
            # for distinct
            for gram in n_grams:
                all_ngram[gram] += 1
            # for repetition
            for gram in set(n_grams):
                if n_grams.count(gram) >= repetition_times:
                    repetition_count += 1
                    break
        metrics[f"repetition-{gram_n}"] = "%.4f" % (repetition_count / float(len(gen_lines)))
        metrics[f"distinct-{gram_n}"] = "%.4f" % (len(all_ngram) / float(all_ngram_num))
    return metrics

