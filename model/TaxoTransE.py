"""
Consider entity from Taxonomy as neighbour nodes to aggregate
Author: Jiaying Lu
Create Date: Jun 7, 2021
"""

import random

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common.util import pad_sequence_to_length


class TaxoTransE(nn.Module):
    def __init__(self, ent_count: int, rel_count: int, taxo_dict: dict,
                 norm: int = 1, dim: int = 100, max_taxo_neigh: int = 8):
        super(TaxoTransE, self).__init__()
        self.ent_count = ent_count
        self.rel_count = rel_count
        self.taxo_dict = taxo_dict   # {'p': {}, 'c': {}}
        self.norm = norm    # norm for (h+r-t), could be 1 or 2
        self.dim = dim
        self.max_taxo_neigh = max_taxo_neigh
        # init emb
        uniform_range = 6 / (self.dim**0.5)
        self.ent_emb = nn.Embedding(num_embeddings=self.ent_count,
                                    embedding_dim=self.dim,
                                    padding_idx=0)
        self.ent_emb.weight.data.uniform_(-uniform_range, uniform_range)
        self.rel_emb = nn.Embedding(num_embeddings=self.rel_count,
                                    embedding_dim=self.dim)
        self.rel_emb.weight.data.uniform_(-uniform_range, uniform_range)

    def _aggregate_over_taxo(self, ents: th.LongTensor) -> th.FloatTensor:
        """
        Args:
            ents: shape=(batch,)
        """
        # current just take mean of parents, children, and self
        neighs = []
        lens = []
        for ent in ents:
            p = self.taxo_dict['p'][ent.item()]
            c = self.taxo_dict['c'][ent.item()]
            neigh = list(p.union(c))
            if len(neigh) > self.max_taxo_neigh:
                neigh = random.sample(neigh, self.max_taxo_neigh)
            neigh = [ent.item()] + neigh
            neighs.append(neigh)
            lens.append(len(neigh))
        max_len = max(lens)
        lens = ents.new_tensor(lens).unsqueeze(1)   # B*1
        neighs = [pad_sequence_to_length(_, max_len) for _ in neighs]
        neighs = ents.new_tensor(neighs)   # B*max_len
        ent_embs = self.ent_emb(neighs).sum(dim=1)    # B*max_len*dim -> B*dim
        ent_embs = th.div(ent_embs, lens.repeat(1, ent_embs.size(1)))    # B*dim
        return ent_embs

    def _cal_distance(self, h: th.FloatTensor, r: th.FloatTensor, t: th.FloatTensor) -> th.FloatTensor:
        """
        Args:
            h, r, t: shape=(batch, dim)
        """
        h = F.normalize(h, p=2, dim=1)    # constraint that L2-norm of emb is 1
        r = F.normalize(r, p=2, dim=1)
        t = F.normalize(t, p=2, dim=1)
        score = (h + r - t).norm(p=self.norm, dim=1)   # (batch,)
        return score

    def forward(self, triples: th.LongTensor) -> th.FloatTensor:
        """
        Args:
            triples: shape=(batch,3)
        """
        h = self._aggregate_over_taxo(triples[:, 0])    # (batch,dim)
        r = self.rel_emb(triples[:, 1])
        t = self._aggregate_over_taxo(triples[:, 2])    # (batch,dim)
        return self._cal_distance(h, r, t)

    def predict(self, triples: th.LongTensor, all_ent_embs: th.FloatTensor) -> th.FloatTensor:
        """
        Args:
            triples: shape=(batch*ent_c, 3)
            all_ent_embs: shape=(ent_c, dim)
        """
        h = all_ent_embs[triples[:, 0]]   # (batch*ent_c, dim)
        r = self.rel_emb(triples[:, 1])
        t = all_ent_embs[triples[:, 2]]
        return self._cal_distance(h, r, t)
