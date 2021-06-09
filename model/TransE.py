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


def cal_metrics(preds: th.Tensor, batch_h: th.Tensor, batch_r: th.Tensor, batch_t: th.Tensor, is_tail_preds: bool, known_triples_map: dict) -> Tuple[int, int, int, float]:
    """
    Args:
        preds: shape=(B,ent_c)
        batch_h,r,t: shape=(B,1)
        known_triples_map: {'h': {(h,r):{t1,t2,t3...}}, 't': {}}
    """
    preds_to_ignore = preds.new_zeros(preds.size())  # non-zero entries for existing triples
    for i in range(preds.size(0)):
        h, r, t = batch_h[i].item(), batch_r[i].item(), batch_t[i].item()
        if is_tail_preds is True:
            ents_to_ignore = list(known_triples_map['h'][(h, r)])
            if t in ents_to_ignore:
                ents_to_ignore.remove(t)
        else:
            ents_to_ignore = list(known_triples_map['t'][(t, r)])
            if h in ents_to_ignore:
                ents_to_ignore.remove(h)
        preds_to_ignore[i][ents_to_ignore] = th.finfo().max
    preds = th.where(preds_to_ignore > 0.0, preds_to_ignore, preds)
    indices = preds.argsort(dim=1)   # B*ent_c, ascending since it is distance
    if is_tail_preds is True:
        ground_truth = batch_t
    else:
        ground_truth = batch_h
    zero_tensor = ground_truth.new_tensor([0])
    one_tensor = ground_truth.new_tensor([1])
    hits_1 = th.where(indices[:, :1] == ground_truth, one_tensor, zero_tensor).sum().item()
    hits_3 = th.where(indices[:, :3] == ground_truth, one_tensor, zero_tensor).sum().item()
    hits_10 = th.where(indices[:, :10] == ground_truth, one_tensor, zero_tensor).sum().item()
    mrr = (1.0 / (indices == ground_truth).nonzero(as_tuple=False)[:, 1].float().add(1.0)).sum().item()
    return hits_1, hits_3, hits_10, mrr


if __name__ == '__main__':
    model = TransE(100, 6)
    triples = th.ones(4, 3).long()
    scores = model.forward(triples)
