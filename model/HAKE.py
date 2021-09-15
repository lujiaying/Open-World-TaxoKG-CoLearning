"""
Adapted from https://github.com/MIRALab-USTC/KGE-HAKE/blob/master/codes/models.py
Author: Jiaying Lu
Create Date: Aug 25, 2021
"""
from typing import Tuple
import math

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .data_loader import BatchType
from .CompGCN import CompGCNLayer

PAD_idx = 0


class HAKE(nn.Module):
    def __init__(self, hid_dim: int, gamma: float, modulus_w: float, phase_w: float):
        """
        For Open-HAKE version
        """
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
        Use one big graph to compute all nodes embedding in one shot
        GCNENcoder aims to learn the projection
        Args:
            comp_opt: implemented ['TransE', 'DistMult']
        """
        super(HAKEGCNEncoder, self).__init__()
        self.dropout = nn.Dropout(p=in_dropout)
        self.gnn1 = CompGCNLayer(in_emb_dim, in_emb_dim, gcn_dropout, comp_opt, use_bn=True)
        self.W_edge1 = nn.Linear(in_emb_dim, in_emb_dim)   # for edges
        self.gnn2 = CompGCNLayer(in_emb_dim, out_emb_dim, gcn_dropout, comp_opt, use_bn=True)
        self.W_edge2 = nn.Linear(in_emb_dim, out_emb_dim)   # for edges

    def forward(self, graph: dgl.DGLGraph, node_embs: th.Tensor, edge_embs: th.Tensor,
                rel_embs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Args:
            graph: 1 big graph
            node_embs: (all_nodes, in_emb)
            edge_embs: (all_edges, in_emb), this is for all edges in graph
            rel_embs: (Batch, in_emb), this is for input triples
        Returns:
            size = (all_nodes, out_emb)
            rel_embs: (Batch, out_emb)
        """
        node_embs = self.encode_graph(graph, node_embs, edge_embs)
        rel_embs = self.encode_relation(rel_embs)
        return node_embs, rel_embs

    def encode_graph(self, graph: dgl.DGLGraph, node_embs: th.Tensor, edge_embs: th.Tensor) -> th.Tensor:
        node_embs = self.dropout(node_embs)
        edge_embs = self.dropout(edge_embs)
        node_embs = self.gnn1(graph, node_embs, edge_embs)
        edge_embs = self.W_edge1(edge_embs)
        node_embs = self.gnn2(graph, F.relu(node_embs), F.relu(edge_embs))
        node_embs = F.relu(node_embs)   # (all_nodes, out_emb)
        return node_embs

    def encode_relation(self, rel_embs: th.Tensor) -> th.Tensor:
        rel_embs = F.relu(self.W_edge1(rel_embs))
        rel_embs = F.relu(self.W_edge2(rel_embs))  # (B , out_emb)
        return rel_embs


class HAKEGCNScorer(nn.Module):
    """
    For HAKE-GCN
    """
    def __init__(self, hid_dim: int, gamma: float, modulus_w: float, phase_w: float,
                 add_rel_bias: bool, do_cart_polar_convt: bool):
        """
        For Open-HAKE version
        Args:
            do_cart_polar_convt: whether to conduct Cartesian to Polar coordinates conversion
        """
        super(HAKEGCNScorer, self).__init__()
        self.hid_dim = hid_dim
        self.gamma = gamma
        # phase, mod for entity (subj, obj)
        self.ent_MLP = nn.Linear(hid_dim, hid_dim*2)   # Linear Transformation
        # phase, mod, bias for relation
        self.add_rel_bias = add_rel_bias
        if add_rel_bias:
            self.rel_MLP = nn.Linear(hid_dim, hid_dim*3)   # Linear Transformation
        else:
            self.rel_MLP = nn.Linear(hid_dim, hid_dim*2)   # Linear Transformation
        self.do_cart_polar_convt = do_cart_polar_convt
        if do_cart_polar_convt:
            self.phase_emb_range = math.pi
        else:
            self.phase_emb_range = 1.0   # we will use Tanh to constrain range
        self.phase_weight = nn.Parameter(th.Tensor([[phase_w * self.phase_emb_range]]))
        self.modulus_weight = nn.Parameter(th.Tensor([[modulus_w]]))

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
        if self.do_cart_polar_convt:
            h = self.convert_cartesian_to_polar(h, is_rel=False)
            r = self.convert_cartesian_to_polar(r, is_rel=True)
            t = self.convert_cartesian_to_polar(t, is_rel=False)
        else:
            h = F.tanh(h)
            r = F.tanh(r)
            t = F.tanh(t)
        return self.func(h, r, t, batch_type)

    def convert_cartesian_to_polar(self, emb: th.Tensor, is_rel: bool) -> th.Tensor:
        if is_rel is True and self.add_rel_bias:
            x, y, bias = th.chunk(emb, 3, dim=2)
            phase = th.atan2(y, x)   # phase ranges [-pi, +pi]
            mod = th.sqrt((x.square() + y.square()))   # mod_i >= 0
            emb = th.cat((phase, mod, bias), dim=2)
        else:
            x, y = th.chunk(emb, 2, dim=2)   # (B, 1/neg_size, h)
            phase = th.atan2(y, x)   # phase ranges [-pi, +pi]
            mod = th.sqrt((x.square() + y.square()))   # mod_i >= 0
            emb = th.cat((phase, mod), dim=2)
        return emb

    def func(self, head: th.tensor, rel: th.tensor, tail: th.tensor, batch_type: int) -> th.tensor:
        phase_head, mod_head = th.chunk(head, 2, dim=2)
        if self.add_rel_bias:
            phase_relation, mod_relation, bias_relation = th.chunk(rel, 3, dim=2)
        else:
            phase_relation, mod_relation = th.chunk(rel, 2, dim=2)
        phase_tail, mod_tail = th.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.phase_emb_range / math.pi)
        phase_relation = phase_relation / (self.phase_emb_range / math.pi)
        phase_tail = phase_tail / (self.phase_emb_range / math.pi)

        if batch_type == BatchType.HEAD_BATCH:
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        if self.add_rel_bias:
            mod_relation = th.abs(mod_relation)
            bias_relation = th.clamp(bias_relation, max=1)
            indicator = (bias_relation < -mod_relation)
            bias_relation[indicator] = -mod_relation[indicator]
            r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)
        else:
            r_score = mod_head * mod_relation - mod_tail

        r_score = th.norm(r_score, dim=2) * self.modulus_weight
        phase_score = th.sum(th.abs(th.sin(phase_score / 2)), dim=2) * self.phase_weight
        return self.gamma - (phase_score + r_score)
