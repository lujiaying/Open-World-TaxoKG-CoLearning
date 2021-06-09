"""
Adapted from https://github.com/mklimasz/TransE-PyTorch/
Author: Jiaying Lu
Create Date: Jun 7, 2021
"""

from typing import Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    def __init__(self, ent_count: int, rel_count: int, norm: int = 1,
                 dim: int = 100):
        super(TransE, self).__init__()
        self.ent_count = ent_count
        self.rel_count = rel_count
        self.norm = norm    # norm for (h+r-t), could be 1 or 2
        self.dim = dim
        # init emb
        uniform_range = 6 / (self.dim**0.5)
        self.ent_emb = nn.Embedding(num_embeddings=self.ent_count,
                                    embedding_dim=self.dim,
                                    padding_idx=0)
        self.ent_emb.weight.data.uniform_(-uniform_range, uniform_range)
        self.rel_emb = nn.Embedding(num_embeddings=self.rel_count,
                                    embedding_dim=self.dim)
        self.rel_emb.weight.data.uniform_(-uniform_range, uniform_range)

    def forward(self, triples: th.LongTensor) -> th.FloatTensor:
        """
        Args:
            triples: shape=(batch,3)
        """
        h = self.ent_emb(triples[:, 0])   # (batch,dim)
        r = self.rel_emb(triples[:, 1])
        t = self.ent_emb(triples[:, 2])
        h = F.normalize(h, p=2, dim=1)    # constraint that L2-norm of emb is 1
        r = F.normalize(r, p=2, dim=1)
        t = F.normalize(t, p=2, dim=1)
        score = (h + r - t).norm(p=self.norm, dim=1)   # (batch,)
        return score


def cal_metrics(preds: th.Tensor, batch_h: th.Tensor, batch_r: th.Tensor, batch_t: th.Tensor, is_tail_preds: bool, known_triples: set) -> Tuple[float, float, float, float]:
    """
    Args:
        preds: shape=(B,ent_c)
        batch_h,r,t: shape=(B,1)
    """
    hits_1 = 0
    hits_3 = 0
    hits_10 = 0
    mrr = 0.0
    indices = preds.argsort(dim=1)   # B*ent_c, ascending since it is distance
    for i in range(preds.size(0)):
        h, r, t = batch_h[i].item(), batch_r[i].item(), batch_t[i].item()
        ranked_list = []    # for one triple
        for index in indices[i]:
            if is_tail_preds is True:
                triple = (h, r, index)
            else:
                triple = (index, r, t)
            if triple in known_triples:
                continue
            ranked_list.append(index)
        if is_tail_preds is True:
            gold_ent = t
        else:
            gold_ent = h
        hits_1 += 1.0 if gold_ent in ranked_list[:1] else 0.0
        hits_3 += 1.0 if gold_ent in ranked_list[:3] else 0.0
        hits_10 += 1.0 if gold_ent in ranked_list[:10] else 0.0
        rank = ranked_list.index(gold_ent) + 1
        mrr += (1.0 / rank)
    return hits_1, hits_3, hits_10, mrr


if __name__ == '__main__':
    model = TransE(100, 6)
    triples = th.ones(4, 3).long()
    scores = model.forward(triples)
