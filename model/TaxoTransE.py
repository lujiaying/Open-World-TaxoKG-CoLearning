"""
Consider entity from Taxonomy as neighbour nodes to aggregate
Author: Jiaying Lu
Create Date: Jun 7, 2021
"""

import random
import time

from scipy import sparse as spsp
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common.util import pad_sequence_to_length

from .data_loader import scipy_sparse2_torch


class TaxoTransE(nn.Module):
    def __init__(self, ent_count: int, rel_count: int, norm: int = 1,
                 dim: int = 100, aggre_type: str = 'mean'):
        """
        Inspired by GraphSage, first aggregate over neighbours
        then concat self and neighbour embeddings
        """
        super(TaxoTransE, self).__init__()
        self.ent_count = ent_count
        self.rel_count = rel_count
        self.norm = norm    # norm for (h+r-t), could be 1 or 2
        self.dim = dim
        self.aggre_type = aggre_type
        # init emb
        uniform_range = 6 / (self.dim**0.5)
        self.ent_emb = nn.Embedding(num_embeddings=self.ent_count,
                                    embedding_dim=self.dim,
                                    padding_idx=0)
        self.ent_emb.weight.data.uniform_(-uniform_range, uniform_range)
        self.rel_emb = nn.Embedding(num_embeddings=self.rel_count,
                                    embedding_dim=self.dim)
        self.rel_emb.weight.data.uniform_(-uniform_range, uniform_range)
        # learnable w for aggregate taxo entities
        # TODO: bilinear to try, bilinear graph neural network with neighbor interactions
        if self.aggre_type == 'mean':
            self.emb_generator = nn.Sequential(
                    nn.Linear(dim * 3, dim, bias=False),
                    nn.GELU(),
                    )
        else:
            print('ERROR create TaxoTransE, invalid aggre_type=%s' % (aggre_type))
            exit(-1)

    def _aggregate_over_taxo(self, ents: th.LongTensor, adj_taxo_p: spsp.csr_matrix,
                             adj_taxo_c: spsp.csr_matrix, is_predict: bool = False) -> th.FloatTensor:
        """
        Args:
            ents: shape=(batch,)
            adj_taxo_p, adj_taxo_c: shape=(ent_c, ent_c)
        """
        device = th.device('cuda') if ents.is_cuda else th.device('cpu')
        batch_size = ents.shape[0]
        ent_count = adj_taxo_p.shape[0]
        if self.aggre_type == 'mean':
            emb_s = self.ent_emb(ents)   # (batch, ent_c)
            # batch_adj_taxo_p = adj_taxo_p.index_select(0, ents)   # (batch, ent_c)
            if is_predict:
                batch_adj_taxo_p = adj_taxo_p
            else:
                batch_adj_taxo_p = adj_taxo_p[ents.cpu().numpy(), :]    # (batch, ent_c)
            batch_adj_taxo_p = scipy_sparse2_torch(batch_adj_taxo_p, (batch_size, ent_count)).to(device)
            emb_p = th.sparse.mm(batch_adj_taxo_p, self.ent_emb.weight)  # (batch, dim)
            # batch_adj_taxo_c = adj_taxo_c.index_select(0, ents)   # (batch, ent_c)
            if is_predict:
                batch_adj_taxo_c = adj_taxo_c
            else:
                batch_adj_taxo_c = adj_taxo_c[ents.cpu().numpy(), :]    # (batch, ent_c)
            batch_adj_taxo_c = scipy_sparse2_torch(batch_adj_taxo_c, (batch_size, ent_count)).to(device)
            emb_c = th.sparse.mm(batch_adj_taxo_c, self.ent_emb.weight)  # (batch, dim)
            emb_ents = th.cat((emb_s, emb_p, emb_c), dim=1)  # (batch, dim*3)
            emb_ents = self.emb_generator(emb_ents)          # (batch, dim)
        return emb_ents

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

    def forward(self, triples: th.LongTensor, adj_taxo_p: th.sparse_coo_tensor,
                adj_taxo_c: th.sparse_coo_tensor) -> th.FloatTensor:
        """
        Args:
            triples: shape=(batch,3)
            adj_taxo_p, adj_taxo_c: shape=(ent_c, ent_c)
        """
        h = self._aggregate_over_taxo(triples[:, 0], adj_taxo_p, adj_taxo_c)    # (batch,dim)
        r = self.rel_emb(triples[:, 1])
        t = self._aggregate_over_taxo(triples[:, 2], adj_taxo_p, adj_taxo_c)    # (batch,dim)
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
