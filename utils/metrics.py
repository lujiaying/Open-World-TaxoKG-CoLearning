import random
from collections import Counter
from typing import Dict

import torch as th


def cal_AP_atk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def cal_reciprocal_rank(actual: list, pred: list) -> float:
    # assume pred must contain element of actual
    for i, p in enumerate(pred):
        if p in actual:
            return 1.0 / (i+1.0)
    return 0.0


def cal_OLP_metrics(preds: th.Tensor, h_mids: list, r_rids: list, t_mids: list,
                    is_tail_preds: bool, all_oie_triples_map: dict, descending: bool = False) -> tuple:
    # adujst pred score for filtered setting
    preds_to_ignore = preds.new_zeros(preds.size())  # non-zero entries for existing triples
    for i in range(preds.size(0)):
        h, r, t = h_mids[i], r_rids[i], t_mids[i]
        if is_tail_preds is True:
            ents_to_ignore = list(all_oie_triples_map['h'][(h, r)])
            if t in ents_to_ignore:
                ents_to_ignore.remove(t)
        else:
            ents_to_ignore = list(all_oie_triples_map['t'][(t, r)])
            if h in ents_to_ignore:
                ents_to_ignore.remove(h)
        preds_to_ignore[i][ents_to_ignore] = 5.0
    fill_value = 0.0 if descending is True else 1e+100
    fill_value = preds.new_tensor(fill_value)
    preds = th.where(preds_to_ignore > 0.0, fill_value, preds)
    preds_idx = preds.argsort(dim=1, descending=descending)   # B*ent_c, ascending since it is distance
    # cal metrics
    """
    # GPU: cost 0.2s
    MRR = []
    Hits10 = []
    Hits30 = []
    Hits50 = []
    for i in range(preds_idx.size(0)):
        gold = [t_mids[i]] if is_tail_preds else [h_mids[i]]
        pred_idx = preds_idx[i].tolist()
        RR = cal_reciprocal_rank(gold, pred_idx)
        MRR.append(RR)
        gold = set(gold)
        H10 = 1.0 if len(gold.intersection(set(pred_idx[:10]))) > 0 else 0.0
        H30 = 1.0 if len(gold.intersection(set(pred_idx[:30]))) > 0 else 0.0
        H50 = 1.0 if len(gold.intersection(set(pred_idx[:50]))) > 0 else 0.0
        Hits10.append(H10)
        Hits30.append(H30)
        Hits50.append(H50)
    """
    # GPU: cost 0.04s
    if is_tail_preds is True:
        ground_truth = preds.new_tensor(t_mids, dtype=th.long).reshape(-1, 1)  # (B,1)
    else:
        ground_truth = preds.new_tensor(h_mids, dtype=th.long).reshape(-1, 1)  # (B,1)
    zero_tensor = ground_truth.new_tensor([0])
    one_tensor = ground_truth.new_tensor([1])
    Hits10 = th.where(preds_idx[:, :10] == ground_truth, one_tensor, zero_tensor).sum().item()
    Hits30 = th.where(preds_idx[:, :30] == ground_truth, one_tensor, zero_tensor).sum().item()
    Hits50 = th.where(preds_idx[:, :50] == ground_truth, one_tensor, zero_tensor).sum().item()
    MRR = (1.0 / (preds_idx == ground_truth).nonzero(as_tuple=False)[:, 1].float().add(1.0)).sum().item()
    return MRR, Hits10, Hits30, Hits50


def cal_OLP_metrics_nontensor(preds: Dict[str, float], h: str, r: str, t: str,
                              is_tail_pred: bool, all_oie_triples_map: dict) -> tuple:
    """
    Non tensor version for calculating metrics
    preds store validity score, the bigger the better.
    """
    if is_tail_pred:
        gold = t
        ents_to_ignore = all_oie_triples_map['h'][(h, r)]
        if t in ents_to_ignore:
            ents_to_ignore.remove(t)
    else:
        gold = h
        ents_to_ignore = all_oie_triples_map['t'][(t, r)]
        if h in ents_to_ignore:
            ents_to_ignore.remove(h)
    for ent in ents_to_ignore:
        preds[ent] = 0.0
    preds = [(k, v) for k, v in preds.items()]
    random.shuffle(preds)
    preds = sorted(preds, key=lambda _: -_[1])
    preds = [_[0] for _ in preds]
    RR = cal_reciprocal_rank([gold], preds)
    h10 = 1.0 if gold in preds[:10] else 0.0
    h30 = 1.0 if gold in preds[:30] else 0.0
    h50 = 1.0 if gold in preds[:50] else 0.0
    return RR, h10, h30, h50


def get_phrase_from_dictid(phr_tids: th.tensor, phr_tlen: int, id2tok: dict) -> str:
    phr_tids = phr_tids.tolist()[:phr_tlen]
    phrase = [id2tok[_] for _ in phr_tids]
    return ' '.join(phrase)


def cal_Shannon_diversity_index(all_preds_idx: list) -> float:
    """
    $H=-\sum p_i ln(p_i)$, where $p_i$ is proportion
    """
    # all_preds_idx size = (n, topk)
    p = Counter()
    for preds_idx in all_preds_idx:
        for idx in preds_idx:
            p[idx] += 1
    p = th.FloatTensor(list(p.values())) / sum(p.values())
    H = (-p * th.log(p)).sum().item()
    return H


def cal_freshness_per_sample(gold: list, pred: list) -> float:
    fresh_cnt = len([_ for _ in pred if _ in gold])
    return 1 - (fresh_cnt / len(pred))


if __name__ == '__main__':
    gold = [1, 2, 5, 51, 52, 53]
    predicted = [1, 2, 3, 4, 5, 6, 7]
    print(cal_AP_atk(gold, predicted))
