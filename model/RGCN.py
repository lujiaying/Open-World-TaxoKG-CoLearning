"""
CompGCN for CGC-OLP-Bench
Author: Anonymous Siamese
Create Date: Sep 24, 2021
"""
from typing import Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

from .data_loader import BatchType

"""
Design:
train dst same as HAKE/HAKEGCN, for negative sampling
data preparation also similar to HAKEGCN

in, out, self-loop three W_r
basis V_b size (num_base, layer_in, layer_out), a_rb size (rel_hdim, num_base)
"""


class RGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_basis: int,
                 self_dropout: float, other_dropout: float):
        super(RGCNLayer, self).__init__()
        self.num_basis = num_basis
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W_rb = nn.Linear(in_dim, num_basis, bias=False)   # rel to a_rb
        self.W_rb_inv = nn.Linear(in_dim, num_basis, bias=False)   # rel to a_rb, for inverse rels
        self.W_basis = nn.Parameter(th.Tensor(num_basis, in_dim*out_dim))
        nn.init.xavier_uniform_(self.W_basis, gain=nn.init.calculate_gain('relu'))
        self.W_self = nn.Linear(in_dim, out_dim)
        self.s_dropout = nn.Dropout(self_dropout)
        self.o_dropout = nn.Dropout(other_dropout)

    def forward(self, graph: dgl.DGLGraph, graph_reverse: dgl.DGLGraph,
                node_embs: th.Tensor, edge_embs: th.Tensor) -> th.Tensor:
        """
        a_rb = self.W_rb(edge_embs)   # (edge, basis)
        W_r = th.mm(a_rb, self.W_basis)  # (edge, in_dim*out_dim)
        graph.ndata['h'] = node_embs
        graph.edata['h'] = W_r.view(-1, self.in_dim, self.out_dim)
        a_rb_inv = self.W_rb_inv(edge_embs)  # (edge, basis)
        W_r_inv = th.mm(a_rb_inv, self.W_basis)  # (edge, in_dim*out_dim)
        graph_reverse.ndata['h'] = node_embs
        graph_reverse.edata['h'] = W_r_inv.view(-1, self.in_dim, self.out_dim)
        graph_reverse.update_all(RGCNLayer.edge_udf, fn.mean('m', 'h'))
        """
        graph.ndata['h'] = node_embs
        graph.edata['h'] = edge_embs
        graph.update_all(self.edge_udf, fn.mean('m', 'h'))
        graph_reverse.ndata['h'] = node_embs
        graph_reverse.edata['h'] = edge_embs
        graph_reverse.update_all(self.edge_inv_udf, fn.mean('m', 'h'))

        h = self.o_dropout(graph.ndata['h']) +\
            self.o_dropout(graph_reverse.ndata['h']) +\
            self.s_dropout(self.W_self(node_embs))
        h = F.relu(h)
        return h

    def edge_udf(self, edges: dgl.udf.EdgeBatch) -> dict:
        # edges.data['h'] size= (B, in)
        W_r = th.mm(self.W_rb(edges.data['h']), self.W_basis)
        # (B, 1, in) B-matmul (B, in, out) -> (B, 1, out)
        m = th.bmm(edges.src['h'].unsqueeze(1), W_r.view(-1, self.in_dim, self.out_dim))
        return {'m': m.squeeze(1)}

    def edge_inv_udf(self, edges: dgl.udf.EdgeBatch) -> dict:
        # edges.data['h'] size= (B, in)
        W_r = th.mm(self.W_rb_inv(edges.data['h']), self.W_basis)
        # (B, 1, in) B-matmul (B, in, out) -> (B, 1, out)
        m = th.bmm(edges.src['h'].unsqueeze(1), W_r.view(-1, self.in_dim, self.out_dim))
        return {'m': m.squeeze(1)}


class RGCN(nn.Module):
    def __init__(self, hid_dim: int, num_basis: int,
                 self_dropout: float = 0.2, other_dropout: float = 0.4):
        """
        dim=200, basis=2.
        Edge dropout: self 0.2, other 0.4.
        """
        super(RGCN, self).__init__()
        self.rgcn1 = RGCNLayer(hid_dim, hid_dim, num_basis, self_dropout, other_dropout)
        self.rgcn2 = RGCNLayer(hid_dim, hid_dim, num_basis, self_dropout, other_dropout)

    def forward(self, graph: dgl.DGLGraph, node_embs: th.Tensor, edge_embs: th.Tensor) -> th.Tensor:
        graph_reverse = dgl.reverse(graph, copy_ndata=False)
        node_embs = self.rgcn1(graph, graph_reverse, node_embs, edge_embs)
        node_embs = self.rgcn2(graph, graph_reverse, node_embs, edge_embs)
        return node_embs


class DistMultDecoder(nn.Module):
    """
    rel embs are updated here.
    To match RGCN design that only regularizes decoder.
    """
    def __init__(self, h_dim: int, dropout: float):
        super(DistMultDecoder, self).__init__()
        self.rel_MLP = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(h_dim, h_dim//2),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(h_dim//2, h_dim),
                nn.ReLU(),
                )

    def forward(self, sample: tuple, batch_type: int) -> th.tensor:
        # sample=(h,r,t), size=(B, h)
        h, r, t = sample
        r = self.rel_MLP(r)
        B = r.size(0)
        if batch_type == BatchType.SINGLE:
            score = (h * r * t).sum(dim=1)  # (B, )
        elif batch_type == BatchType.HEAD_BATCH:
            # h size=(B*neg_size, h)
            neg_size = h.size(0) // B
            r_times_t = (r * t).repeat_interleave(neg_size, dim=0)  # (B*neg, h)
            score = (h * r_times_t).sum(dim=1)  # (B*neg,)
            score = score.view(-1, neg_size)  # (B, neg)
        elif batch_type == BatchType.TAIL_BATCH:
            # t size=(B*neg_size, h)
            neg_size = t.size(0) // B
            h_times_r = (h * r).repeat_interleave(neg_size, dim=0)  # (B*neg, h)
            score = (t * h_times_r).sum(dim=1)    # (B*neg,)
            score = score.view(-1, neg_size)  # (B, neg)
        else:
            raise ValueError('batch_type %s not supported' % (batch_type))
        return score


if __name__ == '__main__':
    in_emb = 16
    rgcn = RGCN(in_emb, 2)
    node_cnt = 14
    edge_cnt = 21
    G = dgl.rand_graph(node_cnt, edge_cnt)
    node_embs = th.rand(node_cnt, in_emb)
    edge_embs = th.rand(edge_cnt, in_emb)
    h = rgcn(G, node_embs, edge_embs)
    print('h size', h.size())
