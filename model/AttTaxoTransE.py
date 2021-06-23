"""
Consider entity from Taxonomy as neighbour nodes to aggregate
attention-based aggregation
Author: Jiaying Lu
Create Date: Jun 23, 2021
"""

import random
import time

from scipy import sparse as spsp
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .data_loader import prepare_batch_taxo_ents


class AttTaxoTransE(nn.Module):
    def __init__(self, ent_count: int, rel_count: int, norm: int = 1,
                 dim: int = 100, attn_dim: int = 10, dropout: float = 0.3,
                 LeakyRELU_slope: float = 0.2, eps: float = 0.01):
        """
        Inspired by GraphSage, first aggregate over neighbours
        then concat self and neighbour embeddings
        """
        super(AttTaxoTransE, self).__init__()
        self.ent_count = ent_count
        self.rel_count = rel_count
        self.norm = norm    # norm for (h+r-t), could be 1 or 2
        self.dim = dim
        # Layers
        self.ent_emb = nn.Embedding(num_embeddings=self.ent_count,
                                    embedding_dim=self.dim,
                                    padding_idx=0)
        self.rel_emb = nn.Embedding(num_embeddings=self.rel_count,
                                    embedding_dim=self.dim)
        self.attn_W_p = nn.Bilinear(dim, dim, attn_dim, bias=False)  # for parents
        self.attn_scorer_p = nn.Sequential(
                nn.linear(attn_dim*2, 1),
                nn.LeakyRELU(negative_slope=LeakyRELU_slope)
                )
        self.attn_W_c = nn.Bilinear(dim, dim, attn_dim, bias=False)  # for children
        self.attn_scorer_c = nn.Sequential(
                nn.linear(attn_dim*2, 1),
                nn.LeakyRELU(negative_slope=LeakyRELU_slope)
                )
        self.dropout = nn.Dropout(p=dropout)
        self.eps = eps    # not learnable
        self.emb_generator = nn.Sequential(
                nn.Linear(dim*3, dim, bias=False),
                nn.RELU(),
                )

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

    def _cal_attn_score(self, emb_s: th.FloatTensor, emb_r: th.FloatTensor, emb_n: th.FloatTensor, is_parent: bool) -> th.FloatTensor:
        """
        Args:
            emb_s: entity self emb, shape=(batch, dim)
            emb_r: rels emb, shape=(batch, dim)
            emb_n: neighs emb, shape=(batch, max_len, dim)
        """
        if is_parent is True:
            attn_W = self.attn_W_p
            attn_scorer = self.attn_scorer_p
        else:
            attn_W = self.attn_W_c
            attn_scorer = self.attn_scorer_c
        h_s = attn_W(emb_r, emb_s)   # (batch, dim_att)
        max_len = emb_n.size(1)
        h_s = h_s.unsqueeze(1).repeat(1, max_len, 1)  # (batch, max_l, dim_att)
        h_n = attn_W(emb_r.unsqueeze(1).repeat(1, max_len, 1), emb_n)   # (batch, max_l, dim_att)
        attn_scores = attn_scorer(th.cat((h_s, h_n), dim=-1))   # (batch, max_l, 1)
        # multiply mask
        return attn_scores

    def _aggregate_over_taxo(self, ents: th.LongTensor, emb_r: th.FloatTensor, taxo_dict: dict) -> th.FloatTensor:
        """
        Args:
            ents: shape=(batch,)
            emb_r: shape=(batch,dim)
        """
        parents, children = prepare_batch_taxo_ents(ents, taxo_dict)   # (batch, max_len)
        emb_s = self.ent_emb(ents)      # self, (batch, dim)
        emb_p = self.ent_emb(parents)   # parents, (batch, max_l, dim)
        emb_c = self.ent_emb(children)  # children, (batch, max_l, dim)
        # calculate attention score

    def forward(self, triples: th.LongTensor, taxo_dict: dict) -> th.FloatTensor:
        """
        Args:
            triples: shape=(batch,3)
        """
        h = triples[:, 0]    # (batch,)
        emb_r = self.rel_emb(triples[:, 1])   # (batch, dim)
