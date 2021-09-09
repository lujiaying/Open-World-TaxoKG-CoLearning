"""
CompGCN for CGC-OLP-Bench
Author: Jiaying Lu
Create Date: Aug 8, 2021
"""
from typing import Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

PAD_idx = 0


class CompGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, comp_opt: str):
        super(CompGCNLayer, self).__init__()
        self.W_O = nn.Linear(in_dim, out_dim)     # for original relations
        self.W_I = nn.Linear(in_dim, out_dim)     # for inverse relations
        self.W_S = nn.Linear(in_dim, out_dim)     # for self-loop
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_dim)
        self.comp_opt = comp_opt

    def forward(self, graphs: dgl.DGLGraph, node_embs: th.Tensor, edge_embs: th.Tensor) -> th.Tensor:
        """
        do not change graphs internal data
        assume no self-loop edge exist
        """
        graphs.ndata['h'] = node_embs
        graphs.edata['h'] = edge_embs
        graphs_reverse = dgl.reverse(graphs, copy_ndata=True, copy_edata=True)
        if self.comp_opt == 'TransE':
            graphs.update_all(fn.u_sub_e('h', 'h', 'm'),
                              fn.mean('m', 'ho'))
            graphs_reverse.update_all(fn.u_sub_e('h', 'h', 'm'),
                                      fn.mean('m', 'hi'))
        elif self.comp_opt == 'DistMult':
            graphs.update_all(fn.u_mul_e('h', 'h', 'm'),
                              fn.mean('m', 'ho'))
            graphs_reverse.update_all(fn.u_mul_e('h', 'h', 'm'),
                                      fn.mean('m', 'hi'))
        else:
            print('invalid comp_opt for CompGCN Layer')
            exit(-1)
        h = 1/3 * self.dropout(self.W_O(graphs.ndata['ho']))\
            + 1/3 * self.dropout(self.W_I(graphs_reverse.ndata['hi']))\
            + 1/3 * self.W_S(node_embs)  # (n_cnt, out_dim)
        h = self.bn(h)
        return h


class CompGCNTransE(nn.Module):
    def __init__(self, in_emb_dim: int, gcn_emb_dim: int, dropout: float, gcn_dropout: float,
                 norm: int, gamma: float, gcn_layer: int, score_func: str = 'TransE'):
        """
        Args:
            score_func: str, used for both composition operator and score function
        """
        super(CompGCNTransE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = norm    # norm for (h+r-t), could be 1 or 2
        self.gamma = gamma  # for TransE score calculate
        self.score_func = score_func
        self.gcn_layer = gcn_layer
        # CompGCN-TransE only 1 GCN layer
        self.gnn1 = CompGCNLayer(in_emb_dim, gcn_emb_dim, gcn_dropout, score_func)
        self.W_rel1 = nn.Linear(in_emb_dim, gcn_emb_dim)   # for edges, no activation
        self.gnn2 = CompGCNLayer(gcn_emb_dim, gcn_emb_dim, gcn_dropout, score_func) if gcn_layer == 2 else None
        self.W_rel2 = nn.Linear(gcn_emb_dim, gcn_emb_dim) if gcn_layer == 2 else None

    def forward(self, graph: dgl.DGLGraph, node_embs: th.Tensor, edge_embs: th.Tensor,
                hids: th.LongTensor, rids: th.LongTensor, tids: th.LongTensor,
                is_head_pred: int) -> tuple:
        hn = self.get_all_node_embs(graph, node_embs, edge_embs)
        hn = self.dropout(hn)
        heads = hn[hids]   # (B, emb)
        tails = hn[tids]   # (B, emb)
        rels = self.W_rel1(edge_embs[rids])
        rels = self.dropout(rels)
        score = self.predict(heads, rels, tails, hn, is_head_pred)
        return hn, score

    def get_all_node_embs(self, graph: dgl.DGLGraph, node_embs: th.Tensor, edge_embs: th.Tensor) -> th.Tensor:
        edge_idices = graph.edata['e_vid']  # (e_cnt,)
        he = edge_embs[edge_idices.squeeze(-1)]  # (e_cnt, emb)
        hn = self.gnn1(graph, node_embs, he)
        hn = F.tanh(hn)
        if self.gcn_layer == 2:
            he = self.W_rel1(he)   # (e_cnt, out_dim)
            hn = self.gnn2(graph, hn, he)
        return hn

    def get_all_edge_embs(self, edge_embs: th.Tensor) -> th.Tensor:
        edge_embs = self.W_rel1(edge_embs)
        if self.gcn_layer == 2:
            edge_embs = self.W_rel2(edge_embs)
        return edge_embs

    def predict(self, h_embs: th.Tensor, r_embs: th.Tensor, t_embs: th.Tensor,
                candidate_embs: th.Tensor, is_head_pred: bool) -> th.Tensor:
        if self.score_func == 'TransE':
            B = h_embs.size(0)
            candidate_cnt = candidate_embs.size(0)
            if is_head_pred == 1:
                # pred on head
                r_minus_t = (r_embs - t_embs).repeat_interleave(candidate_cnt, dim=0)  # (B*cnt, emb)
                dist = (candidate_embs.repeat_interleave(B, dim=0) + r_minus_t).norm(p=self.norm, dim=1)
            else:
                # pred on tail
                h_add_r = (h_embs + r_embs).repeat_interleave(candidate_cnt, dim=0)  # (B*all_n, emb)
                dist = (h_add_r - candidate_embs.repeat_interleave(B, dim=0)).norm(p=self.norm, dim=1)
            score = F.sigmoid(self.gamma - dist.reshape(B, candidate_cnt))
        elif self.score_func == 'DistMult':
            B = h_embs.size(0)
            candidate_cnt = candidate_embs.size(0)
            if is_head_pred == 1:
                # pred on head
                r_times_t = r_embs * t_embs  # (B, emb)
                dist = th.mm(r_times_t, candidate_embs.transpose(1, 0))   # (B, cnt)
            else:
                # pred on tail
                h_times_r = h_embs * r_embs   # (B, emb)
                dist = th.mm(h_times_r, candidate_embs.transpose(1, 0))   # (B, cnt)
            score = F.sigmoid(dist)
        else:
            print('invalid score_func when predict on CompGCN')
            exit(-1)
        return score
