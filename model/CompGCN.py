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
    def __init__(self, in_dim: int, out_dim: int):
        super(CompGCNLayer, self).__init__()
        self.W_O = nn.Linear(in_dim, out_dim)     # for original relations
        self.W_I = nn.Linear(in_dim, out_dim)     # for inverse relations
        self.W_S = nn.Linear(in_dim, out_dim)     # for self-loop
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, graphs: dgl.DGLGraph, node_embs: th.Tensor, edge_embs: th.Tensor) -> th.Tensor:
        """
        do not change graphs internal data
        assume no self-loop edge exist
        """
        graphs.ndata['h'] = node_embs
        graphs.edata['h'] = edge_embs
        graphs_reverse = dgl.reverse(graphs, copy_ndata=True, copy_edata=True)
        graphs.update_all(fn.u_sub_e('h', 'h', 'm'),
                          fn.mean('m', 'ho'))
        graphs_reverse.update_all(fn.u_sub_e('h', 'h', 'm'),
                                  fn.mean('m', 'hi'))
        h = 1/3 * self.W_O(graphs.ndata['ho']) + 1/3 * self.W_I(graphs_reverse.ndata['hi'])\
            + 1/3 * self.W_S(node_embs)  # (n_cnt, out_dim)
        return self.bn(h)


class CompGCNTransE(nn.Module):
    def __init__(self, emb_dim: int, dropout: float, norm: int, gamma: float):
        super(CompGCNTransE, self).__init__()
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(p=dropout)
        self.norm = norm    # norm for (h+r-t), could be 1 or 2
        self.gamma = gamma  # for score calculate
        self.gnn1 = CompGCNLayer(emb_dim, emb_dim//4)
        self.W_rel1 = nn.Linear(emb_dim, emb_dim//4)   # for edges
        self.gnn2 = CompGCNLayer(emb_dim//4, emb_dim)
        self.W_rel2 = nn.Linear(emb_dim//4, emb_dim)   # for edges

    def forward(self, graph: dgl.DGLGraph, node_embs: th.Tensor, edge_embs: th.Tensor,
                is_update_g_embs: bool, hids: th.LongTensor, rids: th.LongTensor,
                tids: th.LongTensor, is_head_pred: int) -> tuple:
        if is_update_g_embs:
            hn = self.get_all_node_embs(graph, node_embs, edge_embs)
        else:
            hn = self.dropout(node_embs)   # use embs from previous batch computed version
        heads = hn[hids]   # (B, emb)
        tails = hn[tids]   # (B, emb)
        rels = self.W_rel1(self.dropout(edge_embs[rids]))
        rels = self.W_rel2(F.relu(rels))   # (B, emb)
        score = self.predict(self.dropout(heads), self.dropout(rels), self.dropout(tails),
                             self.dropout(hn), is_head_pred)
        return hn, score

    def get_all_node_embs(self, graph: dgl.DGLGraph, node_embs: th.Tensor, edge_embs: th.Tensor) -> th.Tensor:
        hn = self.dropout(node_embs)    # here node_embs are output of token_encoder
        he = self.dropout(edge_embs)    # (all_e_cnt, emb)
        edge_idices = graph.edata['e_vid']  # (e_cnt,)
        he = he[edge_idices.squeeze(-1)]  # (e_cnt, emb)
        hn = self.gnn1(graph, hn, he)
        he = self.W_rel1(he)   # (e_cnt, out_dim)
        hn = F.tanh(hn)
        he = F.tanh(he)
        hn = self.gnn2(graph, self.dropout(hn), self.dropout(he))   # (n_cnt, emb)
        # he = self.W_rel2(he)
        return F.tanh(hn)

    def get_all_edge_embs(self, edge_embs: th.Tensor) -> th.Tensor:
        edge_embs = self.dropout(edge_embs)
        edge_embs = self.W_rel1(edge_embs)
        edge_embs = self.W_rel2(F.relu(edge_embs))
        return F.tanh(edge_embs)

    def predict(self, h_embs: th.Tensor, r_embs: th.Tensor, t_embs: th.Tensor,
                candidate_embs: th.Tensor, is_head_pred: bool) -> th.Tensor:
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
        return score
