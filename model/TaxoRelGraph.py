"""
Infer CGC by aggregate relation(bi-direction)
Infer OLP by aggregate taxonomy
Author: Jiaying Lu
Create Date: Jul 15, 2021

Notes:
aggregate over relation: https://arxiv.org/pdf/1911.03082.pdf
k-hop ego local graph of entity `e` for k-layer COMPGCN
h(e/c/r) = mean(emb(e_t1, e_t2, ...))
score func for entity `e` and concept `c`: sigmoid(COMPGCN(h(e))^T * MLP(h(c)))
"""
from typing import Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

PAD_idx = 0


class TokenEncoder(nn.Module):
    def __init__(self, tok_cnt: int, emb_dim: int):
        super(TokenEncoder, self).__init__()
        self.tok_cnt = tok_cnt
        self.emb_dim = emb_dim
        # init emb
        uniform_range = 6 / (self.emb_dim**0.5)
        self.encoder = nn.Embedding(num_embeddings=tok_cnt,
                                    embedding_dim=emb_dim,
                                    padding_idx=PAD_idx)
        self.encoder.weight.data.uniform_(-uniform_range, uniform_range)

    def forward(self, tok_batch: th.LongTensor, tok_lens: th.LongTensor) -> th.tensor:
        """
        Args:
            tok_batch: shape=(B, L)
            tok_lens: shape=(B,)
        """
        tok_embs = self.encoder(tok_batch)  # (B, L, emb_d)
        tok_embs = tok_embs.sum(dim=1)   # (B, emb_d)
        tok_embs = tok_embs / tok_lens.view(-1, 1)  # (B, emb_d)
        return tok_embs


class CompGCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, is_semantic_edge: bool = True):
        super(CompGCN, self).__init__()
        self.is_semantic_edge = is_semantic_edge
        self.W_O = nn.Linear(in_dim, out_dim)   # for original relations
        self.W_I = nn.Linear(in_dim, out_dim)   # for inverse relations
        # note: self-loop will be mutliplied by both W_I and W_O
        if self.is_semantic_edge:
            self.W_rel = nn.Linear(in_dim, out_dim)

    def forward(self, graphs: dgl.DGLGraph, node_embs: th.Tensor, edge_embs: th.Tensor):
        """
        do not change graphs internal data
        """
        graphs.ndata['h'] = node_embs
        if self.is_semantic_edge:
            graphs.edata['h'] = edge_embs
        graphs_reverse = dgl.reverse(graphs, copy_ndata=True, copy_edata=True)
        if self.is_semantic_edge:
            graphs.update_all(fn.u_sub_e('h', 'h', 'm'),
                              fn.sum('m', 'ho'))
            graphs_reverse.update_all(fn.u_sub_e('h', 'h', 'm'),
                                      fn.sum('m', 'hi'))
        else:
            graphs.update_all(fn.copy_u('h', 'm'),
                              fn.mean('m', 'ho'))
            graphs_reverse.update_all(fn.copy_u('h', 'm'),
                                      fn.mean('m', 'hi'))
        h = self.W_O(graphs.ndata['ho']) + self.W_I(graphs_reverse.ndata['hi'])  # (n_cnt, out_dim)
        if self.is_semantic_edge:
            he = self.W_rel(graphs.edata['h'])   # (e_cnt, out_dim)
        else:
            he = None
        return h, he


class TaxoRelCGC(nn.Module):
    def __init__(self, emb_dim: int, dropout: float, g_readout: str):
        super(TaxoRelCGC, self).__init__()
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(p=dropout)
        self.g_readout = g_readout
        self.gnn1 = CompGCN(emb_dim, emb_dim//4)
        self.gnn2 = CompGCN(emb_dim//4, emb_dim)
        self.cep_encoder = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU()
                )

    def forward(self, graphs: dgl.DGLGraph, node_embs: th.Tensor, edge_embs: th.Tensor, cep_embs: th.Tensor):
        """
        Args:
            node_embs: (n_cnt, emb_d)
            edge_embs: (e_cnt, emb_d)
            cep_embs:  (all_c_cnt, emb_d)
        """
        node_embs = self.dropout(node_embs)
        edge_embs = self.dropout(edge_embs)
        hn, he = self.gnn1(graphs, node_embs, edge_embs)
        hn = F.relu(hn)
        he = F.relu(he)
        hn, he = self.gnn2(graphs, hn, he)   # (n/e_cnt, emb_d)
        hn = F.relu(hn)    # (n_cnt, emb_d)
        graphs.ndata['h'] = hn
        hg = dgl.readout_nodes(graphs, 'h', op=self.g_readout)  # (batch, emb_d)
        cep_embs = self.cep_encoder(cep_embs)        # (all_c_cnt, emb_d)
        logits = th.matmul(hg, cep_embs.transpose(0, 1))   # (batch, all_c_cnt)
        return logits


class TaxoRelOLP(nn.Module):
    def __init__(self, emb_dim: int, dropout: float, g_readout: str, norm: int):
        super(TaxoRelOLP, self).__init__()
        self.emb_dim = emb_dim
        self.norm = norm    # norm for (h+r-t), could be 1 or 2
        self.dropout = nn.Dropout(p=dropout)
        self.g_readout = g_readout
        self.gnn1 = CompGCN(emb_dim, emb_dim//4, is_semantic_edge=False)
        self.gnn2 = CompGCN(emb_dim//4, emb_dim, is_semantic_edge=False)
        self.rel_encoder = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU()
                )

    def _compute_graph_emb(self, bg: dgl.DGLGraph, node_embs: th.Tensor) -> th.Tensor:
        node_embs = self.dropout(node_embs)
        _ = None
        hn, _ = self.gnn1(bg, node_embs, _)
        hn = F.relu(hn)
        hn, _ = self.gnn2(bg, hn, _)
        hn = F.relu(hn)
        bg.ndata['h'] = hn
        hg = dgl.readout_nodes(bg, 'h', op=self.g_readout)  # (batch, emb_d)
        return hg

    def _sample_batch_negative_triples(self, h_embs: th.Tensor, t_embs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        batch_size = h_embs.size(0)
        head_or_tail = th.randint(high=2, size=(batch_size,))
        random_rows = th.randperm(batch_size)
        corrupt_hidx = th.where(head_or_tail == 1, random_rows, th.arange(batch_size))
        corrupt_tidx = th.where(head_or_tail == 0, random_rows, th.arange(batch_size))
        corrupt_h_embs = h_embs[corrupt_hidx]
        corrupt_t_embs = t_embs[corrupt_tidx]
        return corrupt_h_embs, corrupt_t_embs

    def _cal_distance(self, h: th.FloatTensor, r: th.FloatTensor, t: th.FloatTensor) -> th.FloatTensor:
        """
        Args:
            h, r, t: shape=(batch, dim)
        """
        # h = F.normalize(h, p=2, dim=1)    # constraint that L2-norm of emb is 1
        # r = F.normalize(r, p=2, dim=1)
        # t = F.normalize(t, p=2, dim=1)
        score = (h + r - t).norm(p=self.norm, dim=1)   # (batch,)
        return score

    def forward(self, subj_bg: dgl.DGLGraph, subj_node_embs: th.Tensor, rel_tok_embs: th.Tensor,
                obj_bg: dgl.DGLGraph, obj_node_embs: th.Tensor):
        h_embs = self._compute_graph_emb(subj_bg, subj_node_embs)   # (batch, emb_d)
        t_embs = self._compute_graph_emb(obj_bg, obj_node_embs)      # (batch, emb_d)
        r_embs = self.rel_encoder(rel_tok_embs)
        corrupt_h_embs, corrupt_t_embs = self._sample_batch_negative_triples(h_embs, t_embs)
        pos_scores = self._cal_distance(h_embs, r_embs, t_embs)
        neg_scores = self._cal_distance(corrupt_h_embs, r_embs, corrupt_t_embs)
        return pos_scores, neg_scores

    def test_tail_pred(self, h_embs: th.tensor, r_tok_embs: th.tensor,
                       all_ment_embs: th.tensor) -> th.tensor:
        ment_cnt = all_ment_embs.size(0)
        B = h_embs.size(0)
        r_embs = self.rel_encoder(r_tok_embs)   # (B, emb_d)
        h_embs = h_embs.repeat_interleave(ment_cnt, dim=0)  # (B*m_cnt, emb_d)
        r_embs = r_embs.repeat_interleave(ment_cnt, dim=0)  # (B*m_cnt, emb_d)
        all_ment_embs = all_ment_embs.repeat_interleave(B, dim=0)  # (B*m_cnt, emb_d)
        score = self._cal_distance(h_embs, r_embs, all_ment_embs)  # (B*m_cnt)
        return score.reshape(B, ment_cnt)

    def test_head_pred(self, t_embs: th.tensor, r_tok_embs: th.tensor,
                       all_ment_embs: th.tensor) -> th.tensor:
        ment_cnt = all_ment_embs.size(0)
        B = t_embs.size(0)
        r_embs = self.rel_encoder(r_tok_embs)   # (B, emb_d)
        t_embs = t_embs.repeat_interleave(ment_cnt, dim=0)  # (B*m_cnt, emb_d)
        r_embs = r_embs.repeat_interleave(ment_cnt, dim=0)  # (B*m_cnt, emb_d)
        all_ment_embs = all_ment_embs.repeat_interleave(B, dim=0)  # (B*m_cnt, emb_d)
        score = self._cal_distance(all_ment_embs, r_embs, t_embs)  # (B*m_cnt)
        return score.reshape(B, ment_cnt)
