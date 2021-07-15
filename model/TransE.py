"""
Adapted from https://github.com/mklimasz/TransE-PyTorch/,
  and Can We Predict New Facts with Open Knowledge Graph Embeddings?
Author: Jiaying Lu
Create Date: Jul 15, 2021
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class OpenTransE(nn.Module):
    def __init__(self, tok_count: int, emb_dim: int, norm: int):
        super(OpenTransE, self).__init__()
        self.tok_count = tok_count
        self.norm = norm    # norm for (h+r-t), could be 1 or 2
        self.emb_dim = emb_dim
        self.lstm_layer = 1
        # init emb
        uniform_range = 6 / (self.emb_dim**0.5)
        self.tok_emb = nn.Embedding(num_embeddings=self.tok_count,
                                    embedding_dim=self.emb_dim,
                                    padding_idx=0)
        self.tok_emb.weight.data.uniform_(-uniform_range, uniform_range)
        # funcs for tok_emb -> mention_emb, rel_emb
        self.mention_func = nn.LSTM(input_size=self.emb_dim,
                                    hidden_size=self.emb_dim,
                                    num_layers=self.lstm_layer,
                                    batch_first=True)
        self.rel_func = nn.LSTM(input_size=self.emb_dim,
                                hidden_size=self.emb_dim,
                                num_layers=self.lstm_layer,
                                batch_first=True)

    def _get_composition_emb(self, tok_batch: th.LongTensor, lens: th.LongTensor, func: nn.Module) -> nn.Tensor:
        h_embs = self.tok_emb(tok_batch)   # (B, L, emb_dim)
        h_embs = pack_padded_sequence(h_embs, lens, batch_first=True, enforce_sorted=False)
        h_embs, (h, c) = func(h_embs)
        h_embs, _ = pad_packed_sequence(h_embs, batch_first=True)  # (B, L, emb_dim)
        h_embs = h_embs[th.arange(h_embs.size(0)), lens-1]  # (B, emb_dim)
        return h_embs

    def forward(self, h_batch: th.LongTensor, r_batch: th.LongTensor, t_batch: th.LongTensor, h_lens: th.LongTensor, r_lens: th.LongTensor, t_lens: th.LongTensor):
        h_embs = self._get_composition_emb(h_batch, h_lens, self.mention_func)
        r_embs = self._get_composition_emb(r_batch, r_lens, self.rel_func)
        t_embs = self._get_composition_emb(t_batch, t_lens, self.mention_func)
        # batch negative sampling
