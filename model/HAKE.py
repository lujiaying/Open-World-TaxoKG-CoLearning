"""
Adapted from https://github.com/MIRALab-USTC/KGE-HAKE/blob/master/codes/models.py
Author: Jiaying Lu
Create Date: Aug 25, 2021
"""
from typing import Tuple
import math
from collections import OrderedDict

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

from .data_loader import BatchType
from .CompGCN import CompGCNLayer

PAD_idx = 0
Epsilon = 1e-10


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


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, in_dropout: float):
        super(GCNLayer, self).__init__()
        self.W_O = nn.Linear(in_dim, in_dim)     # for original relations
        self.W_I = nn.Linear(in_dim, in_dim)     # for inverse relations
        self.W_S = nn.Linear(in_dim, in_dim)     # for self-loop
        self.MLP = nn.Sequential(OrderedDict([
                ('dropout', nn.Dropout(p=in_dropout)),
                ('dense', nn.Linear(in_dim*2, out_dim)),
                ('prelu', nn.PReLU())
                ]))

    def forward(self, graph: dgl.DGLGraph,
                node_embs: th.Tensor, edge_embs: th.Tensor) -> th.Tensor:
        # edge, neigh composition in Cartesian Coordinates
        # aggregation in Polar COordinates
        # graph.update_all(fn.u_sub_e('h', 'h', 'm'),
        #                  GCNLayer.node_udf)
        # graph_reverse.update_all(fn.u_sub_e('h', 'h', 'm'),
        #                          GCNLayer.node_udf)
        graph.ndata['h'] = node_embs
        graph.edata['h'] = edge_embs
        graph.update_all(fn.u_sub_e('h', 'h', 'm'),
                         fn.mean('m', 'h'))
        h_neigh = self.W_O(graph.ndata['h'])   # h for original edges
        graph.ndata.pop('h')
        graph_reverse = dgl.reverse(graph, copy_ndata=False)
        graph_reverse.ndata['h'] = node_embs
        graph_reverse.edata['h'] = edge_embs
        graph_reverse.update_all(fn.u_sub_e('h', 'h', 'm'),
                                 fn.mean('m', 'h'))
        h_neigh = h_neigh + self.W_I(graph_reverse.ndata['h'])   # h for inverse edges
        graph_reverse.ndata.pop('h')
        hs = self.W_S(node_embs)
        # h = (ho + hi + hs) / 3.0
        h = self.MLP(th.cat((hs, h_neigh), dim=1))
        return h

    @staticmethod
    def node_udf(nodes: dgl.udf.NodeBatch) -> dict:
        phase, mod = GCNLayer.convert_cartesian_to_polar(nodes.mailbox['m'], False)
        # phase, mod size = (N, D, h/2)
        # circular mean for phase
        phase = th.atan2(th.sin(phase).sum(dim=1), th.cos(phase).sum(dim=1))  # (N, h/2)
        # geometric mean for modulus
        mod = th.exp(th.log(mod+Epsilon).mean(dim=1))   # (N, h/2)
        # arithmetic mean for modulus
        # mod = mod.mean(dim=1)
        return {'h': GCNLayer.convert_polar_to_cartesian(th.cat((phase, mod), dim=-1), True)}

    @staticmethod
    def convert_cartesian_to_polar(emb: th.Tensor, do_cat: bool) -> th.Tensor:
        x, y = th.chunk(emb, 2, dim=-1)   # (B, h/2)
        phase = th.atan2(y, x)   # phase ranges [-pi, +pi]
        mod = th.sqrt((x.square() + y.square()))   # mod_i >= 0
        if do_cat:
            emb = th.cat((phase, mod), dim=-1)
            return emb
        else:
            return (phase, mod)

    @staticmethod
    def convert_polar_to_cartesian(emb: th.Tensor, do_cat: bool) -> th.Tensor:
        phase, mod = th.chunk(emb, 2, dim=-1)   # (B, h/2)
        x = mod * th.cos(phase)
        y = mod * th.sin(phase)
        if do_cat:
            emb = th.cat((x, y), dim=-1)
            return emb
        else:
            return (x, y)


class HAKEGCNEncoder(nn.Module):
    def __init__(self, in_emb_dim: int, in_dropout: float, out_emb_dim: int,
                 gcn_layer: int = 2):
        """
        Use one big graph to compute all nodes embedding in one shot
        GCNENcoder aims to learn the projection
        Args:
            comp_opt: implemented ['TransE', 'DistMult']
        """
        super(HAKEGCNEncoder, self).__init__()
        self.ent_MLP = nn.Sequential(OrderedDict([
                ('dropout', nn.Dropout(p=in_dropout)),
                ('dense', nn.Linear(in_emb_dim, out_emb_dim)),
                ('prelu', nn.PReLU())
                ]))
        self.rel_MLP = nn.Sequential(OrderedDict([
                ('dropout', nn.Dropout(p=in_dropout)),
                ('dense', nn.Linear(in_emb_dim, out_emb_dim)),
                ('prelu', nn.PReLU())
                ]))
        self.gnn1 = GCNLayer(out_emb_dim, out_emb_dim, in_dropout)
        self.W_edge1 = nn.Sequential(OrderedDict([
                ('dropout', nn.Dropout(p=in_dropout)),
                ('dense', nn.Linear(out_emb_dim, out_emb_dim)),
                ('prelu', nn.PReLU())
                ]))
        self.gcn_layer = gcn_layer
        if gcn_layer == 2:
            self.gnn2 = GCNLayer(out_emb_dim, out_emb_dim, in_dropout)
            self.W_edge2 = nn.Sequential(OrderedDict([
                    ('dropout', nn.Dropout(p=in_dropout)),
                    ('dense', nn.Linear(out_emb_dim, out_emb_dim)),
                    ('prelu', nn.PReLU())
                    ]))

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
        node_embs = self.ent_MLP(node_embs)
        edge_embs = self.rel_MLP(edge_embs)
        node_embs = self.gnn1(graph, node_embs, edge_embs)
        if self.gcn_layer == 2:
            edge_embs = self.W_edge1(edge_embs)
            node_embs = self.gnn2(graph, node_embs, edge_embs)  # (all_n, out_emb)
        return node_embs

    def encode_relation(self, rel_embs: th.Tensor) -> th.Tensor:
        rel_embs = self.rel_MLP(rel_embs)
        rel_embs = self.W_edge1(rel_embs)  # (B, out_emb)
        if self.gcn_layer == 2:
            rel_embs = self.W_edge2(rel_embs)  # (B, out_emb)
        return rel_embs


class HAKEGCNScorer(nn.Module):
    """
    For HAKE-GCN
    """
    def __init__(self, hid_dim: int, gamma: float, modulus_w: float, phase_w: float):
        super(HAKEGCNScorer, self).__init__()
        self.gamma = gamma
        self.phase_emb_range = math.pi
        self.phase_weight = nn.Parameter(th.Tensor([[phase_w * self.phase_emb_range]]))
        self.modulus_weight = nn.Parameter(th.Tensor([[modulus_w]]))

    def forward(self, sample: tuple, batch_type: int) -> th.tensor:
        # sample=(h,r,t), size=(B, h)
        h, r, t = sample
        if batch_type == BatchType.SINGLE:
            h = h.unsqueeze(1)   # (B, 1, h)
            r = r.unsqueeze(1)
            t = t.unsqueeze(1)
        elif batch_type == BatchType.HEAD_BATCH:
            # h size=(B, neg_size, h)
            r = r.unsqueeze(1)  # (B, 1, h)
            t = t.unsqueeze(1)  # (B, 1, h)
        elif batch_type == BatchType.TAIL_BATCH:
            h = h.unsqueeze(1)   # (B, 1, h)
            r = r.unsqueeze(1)   # (B, 1, h)
            # t size=(B, neg_size, h)
        else:
            raise ValueError('batch_type %s not supported' % (batch_type))
        h = self.convert_cartesian_to_polar(h)
        r = self.convert_cartesian_to_polar(r)
        t = self.convert_cartesian_to_polar(t)
        return self.func(h, r, t, batch_type)

    def convert_cartesian_to_polar(self, emb: th.Tensor) -> th.Tensor:
        x, y = th.chunk(emb, 2, dim=2)   # (B, 1/neg_size, h/2)
        phase = th.atan2(y, x)   # phase ranges [-pi, +pi]
        mod = th.sqrt((x.square() + y.square()))   # mod_i >= 0
        emb = th.cat((phase, mod), dim=2)
        return emb

    def func(self, head: th.tensor, rel: th.tensor, tail: th.tensor, batch_type: int) -> th.tensor:
        phase_head, mod_head = th.chunk(head, 2, dim=2)
        phase_relation, mod_relation = th.chunk(rel, 2, dim=2)
        phase_tail, mod_tail = th.chunk(tail, 2, dim=2)
        """
        phase_head = phase_head / (self.phase_emb_range / math.pi)
        phase_relation = phase_relation / (self.phase_emb_range / math.pi)
        phase_tail = phase_tail / (self.phase_emb_range / math.pi)
        """
        if batch_type == BatchType.HEAD_BATCH:
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail
        r_score = mod_head * mod_relation - mod_tail

        r_score = th.norm(r_score, dim=2) * self.modulus_weight
        # phase_score = th.sum(th.abs(th.sin(phase_score / 2)), dim=2) * self.phase_weight
        # TODO: not divide 2, change to L2norm
        phase_score = th.sum(th.abs(th.sin(phase_score)), dim=2) * self.phase_weight
        return self.gamma - (phase_score + r_score)
