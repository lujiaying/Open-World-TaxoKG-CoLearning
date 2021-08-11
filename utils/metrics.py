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
    preds = th.where(preds_to_ignore > 0.0, preds_to_ignore, preds)
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


if __name__ == '__main__':
    gold = [1, 2, 5, 51, 52, 53]
    predicted = [1, 2, 3, 4, 5, 6, 7]
    print(cal_AP_atk(gold, predicted))
