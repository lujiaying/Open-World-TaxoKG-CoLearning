"""
Adapted from https://github.com/MIRALab-USTC/KGE-HAKE/blob/master/codes/models.py
Author: Jiaying Lu
Create Date: Aug 25, 2021
"""
from typing import Tuple

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .data_loader import BatchType
from .CompGCN import CompGCNLayer

PAD_idx = 0


class HAKE(nn.Module):
    def __init__(self, hid_dim: int, gamma: float, modulus_w: float, phase_w: float):
        super(HAKE, self).__init__()
        self.hid_dim = hid_dim
        self.gamma = gamma
        # phase, mod
        self.ent_MLP = nn.Sequential(
                nn.Linear(hid_dim, hid_dim*2),
                nn.Tanh(),
                )
        # phase, mod, bias
        self.rel_MLP = nn.Sequential(
                nn.Linear(hid_dim, hid_dim*3),
                nn.Tanh(),
                )
        self.embedding_range = 1.0   # as we use Tanh
        self.phase_weight = nn.Parameter(th.Tensor([[phase_w * self.embedding_range]]))
        self.modulus_weight = nn.Parameter(th.Tensor([[modulus_w]]))
        self.pi = 3.14159262358979323846

    def forward(self, sample: tuple, batch_type: int):
        if batch_type == BatchType.SINGLE:
            # sample=(h,r,t), size=(B, h)
            h, r, t = sample
            h = self.ent_MLP(h).unsqueeze(1)   # (B, 1, 2*h)
            r = self.rel_MLP(r).unsqueeze(1)
            t = self.ent_MLP(t).unsqueeze(1)
        elif batch_type == BatchType.HEAD_BATCH:
            h, r, t = sample
            h = self.ent_MLP(h)   # (B, neg_size, 2*h)
            r = self.rel_MLP(r).unsqueeze(1)
            t = self.ent_MLP(t).unsqueeze(1)  # (B, 1, 2*h)
        elif batch_type == BatchType.TAIL_BATCH:
            h, r, t = sample
            h = self.ent_MLP(h).unsqueeze(1)   # (B, 1, 2*h)
            r = self.rel_MLP(r).unsqueeze(1)
            t = self.ent_MLP(t)  # (B, neg_size, 2*h)
        else:
            raise ValueError('batch_type %s not supported' % (batch_type))
        return self.func(h, r, t, batch_type)

    def func(self, head: th.tensor, rel: th.tensor, tail: th.tensor, batch_type: int) -> th.tensor:
        phase_head, mod_head = th.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = th.chunk(rel, 3, dim=2)
        phase_tail, mod_tail = th.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.embedding_range / self.pi)
        phase_relation = phase_relation / (self.embedding_range / self.pi)
        phase_tail = phase_tail / (self.embedding_range / self.pi)

        if batch_type == BatchType.HEAD_BATCH:
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = th.abs(mod_relation)
        bias_relation = th.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)
        r_score = th.norm(r_score, dim=2) * self.modulus_weight
        phase_score = th.sum(th.abs(th.sin(phase_score / 2)), dim=2) * self.phase_weight
        return self.gamma - (phase_score + r_score)


class HAKEGCNEncoder(nn.Module):
    def __init__(self, in_emb_dim: int, in_dropout: float, out_emb_dim: int, gcn_dropout: float, comp_opt: str):
        """
        Args:
            comp_opt: implemented ['TransE', 'DistMult']
        """
        super(HAKEGCNEncoder, self).__init__()
        self.dropout = nn.Dropout(p=in_dropout)
        self.gnn1 = CompGCNLayer(in_emb_dim, in_emb_dim//2, gcn_dropout, comp_opt, use_bn=False)
        self.W_edge1 = nn.Linear(in_emb_dim, in_emb_dim//2)   # for edges
        self.gnn2 = CompGCNLayer(in_emb_dim//2, out_emb_dim, gcn_dropout, comp_opt, use_bn=False)

    def forward(self, graphs: dgl.DGLGraph, node_embs: th.Tensor, edge_embs: th.Tensor) -> th.Tensor:
        """
        Args:
            graphs: (Batch,)
            node_embs: (all_nodes, in_emb)
            edge_embs: (all_edges, in_emb)
        Returns:
            size = (Batch, gcn_emb)
        """
        node_embs = self.dropout(node_embs)
        edge_embs = self.dropout(edge_embs)
        node_embs = self.gnn1(graphs, node_embs, edge_embs)
        edge_embs = self.W_edge1(edge_embs)
        node_embs = self.gnn2(graphs, F.relu(node_embs), F.relu(edge_embs))
        g_node_cnts = graphs.batch_num_nodes()   # (B, )
        central_nids = [0] + g_node_cnts.cumsum(dim=0).tolist()[:-1]
        return node_embs[central_nids]
