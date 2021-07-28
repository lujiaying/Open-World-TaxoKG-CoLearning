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
    def __init__(self, in_dim: int, out_dim: int):
        super(CompGCN, self).__init__()
        self.W_O = nn.Linear(in_dim, out_dim)   # for original relations
        self.W_I = nn.Linear(in_dim, out_dim)   # for inverse relations
        # note: self-loop will be mutliplied by both W_I and W_O
        self.W_rel = nn.Linear(in_dim, out_dim)

    def forward(self, graphs: dgl.DGLGraph, node_embs: th.Tensor, edge_embs: th.Tensor):
        """
        do not change graphs internal data
        """
        graphs.ndata['h'] = node_embs
        graphs.edata['h'] = edge_embs
        graphs_reverse = dgl.reverse(graphs, copy_ndata=True, copy_edata=True)
        graphs.update_all(fn.v_sub_e('h', 'h', 'm'),
                          fn.sum('m', 'ho'))
        graphs_reverse.update_all(fn.v_sub_e('h', 'h', 'm'),
                                  fn.sum('m', 'hi'))
        h = self.W_O(graphs.ndata['ho']) + self.W_I(graphs_reverse.ndata['hi'])  # (n_cnt, out_dim)
        he = self.W_rel(graphs.edata['h'])   # (e_cnt, out_dim)
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
