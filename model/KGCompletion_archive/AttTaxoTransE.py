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
from allennlp.nn.util import get_mask_from_sequence_lengths, masked_softmax

from .data_loader import prepare_batch_taxo_ents


class AttTaxoTransE(nn.Module):
    def __init__(self, ent_count: int, rel_count: int, norm: int = 1,
                 dim: int = 100, attn_dim: int = 32, dropout: float = 0.3,
                 LeakyReLU_slope: float = 0.2, eps: float = 0.01):
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
        # for parents, no bias in GAT paper
        self.attn_scorer_p = nn.Sequential(
                nn.Linear(dim*2, attn_dim, bias=False),
                nn.Linear(attn_dim, 1, bias=False),
                nn.LeakyReLU(negative_slope=LeakyReLU_slope)
                )
        # for children, no bias in GAT paper
        self.attn_scorer_c = nn.Sequential(
                nn.Linear(dim*2, attn_dim, bias=False),
                nn.Linear(attn_dim, 1, bias=False),
                nn.LeakyReLU(negative_slope=LeakyReLU_slope)
                )
        self.eps = eps    # not learnable
        self.emb_generator = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(dim*3, dim, bias=True),
                nn.ReLU(),
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

    def _cal_attn_score(self, emb_s: th.FloatTensor, emb_n: th.FloatTensor,
                        mask: th.BoolTensor, is_parent: bool) -> th.FloatTensor:
        """
        Args:
            emb_s: entity self emb, shape=(batch, dim)
            emb_n: neighs emb, shape=(batch, max_len, dim)
            mask: shape=(batch, max_len)
        """
        if is_parent is True:
            attn_scorer = self.attn_scorer_p
        else:
            attn_scorer = self.attn_scorer_c
        max_len = emb_n.size(1)
        h_s = emb_s.unsqueeze(1).repeat(1, max_len, 1)  # (batch, max_l, dim)
        attn_scores = attn_scorer(th.cat((h_s, emb_n), dim=-1))   # (batch, max_l, 1)
        attn_scores = masked_softmax(attn_scores.squeeze(2), mask, dim=1)  # (batch, max_l)
        return attn_scores

    def _get_r_emb(self, r: th.LongTensor) -> th.FloatTensor:
        return self.rel_emb(r)

    def _aggregate_over_taxo(self, ents: th.LongTensor, taxo_dict: dict) -> th.FloatTensor:
        """
        Args:
            ents: shape=(batch,)
        """
        (parents, lens_p), (children, lens_c) = prepare_batch_taxo_ents(ents, taxo_dict)   # (batch, max_len)
        mask_p = get_mask_from_sequence_lengths(lens_p, parents.size(1))   # (batch, max_l)
        mask_c = get_mask_from_sequence_lengths(lens_c, children.size(1))  # (batch, max_l)
        emb_s = self.ent_emb(ents)      # self, (batch, dim)
        emb_p = self.ent_emb(parents)   # parents, (batch, max_l, dim)
        emb_c = self.ent_emb(children)  # children, (batch, max_l, dim)
        # calculate attention score
        attn_score_p = self._cal_attn_score(emb_s, emb_p, mask_p, True)   # (batch, max_l)
        attn_score_c = self._cal_attn_score(emb_s, emb_c, mask_c, False)  # (batch, max_l)
        emb_p = (attn_score_p.unsqueeze(2).repeat(1, 1, self.dim) * emb_p).sum(dim=1)  # (batch, dim)
        emb_c = (attn_score_c.unsqueeze(2).repeat(1, 1, self.dim) * emb_c).sum(dim=1)  # (batch, dim)
        emb_aggre = th.cat(((1+self.eps)*emb_s, emb_p, emb_c), dim=1)
        emb_aggre = self.emb_generator(emb_aggre)   # (batch, dim)
        return emb_aggre

    def forward(self, triples: th.LongTensor, taxo_dict: dict) -> th.FloatTensor:
        """
        Args:
            triples: shape=(batch,3)
        """
        r = self.rel_emb(triples[:, 1])   # (batch, dim)
        h = self._aggregate_over_taxo(triples[:, 0], taxo_dict)
        t = self._aggregate_over_taxo(triples[:, 2], taxo_dict)
        return self._cal_distance(h, r, t)

    def predict(self, triples: th.LongTensor, all_embs: th.Tensor) -> th.FloatTensor:
        """
        Args:
            triples: shape=(batch*ent_c,3)
            all_embs: shape=(ent_c,dim)
        """
        r = self.rel_emb(triples[:, 1])
        h = all_embs[triples[:, 0], :]
        t = all_embs[triples[:, 2], :]
        return self._cal_distance(h, r, t)
