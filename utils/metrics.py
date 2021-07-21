import numpy as np


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


if __name__ == '__main__':
    gold = [1, 2, 5, 51, 52, 53]
    predicted = [1, 2, 3, 4, 5, 6, 7]
    print(cal_AP_atk(gold, predicted))
