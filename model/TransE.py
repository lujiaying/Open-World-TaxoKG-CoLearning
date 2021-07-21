"""
Adapted from https://github.com/mklimasz/TransE-PyTorch/,
  and Can We Predict New Facts with Open Knowledge Graph Embeddings?
Author: Jiaying Lu
Create Date: Jul 15, 2021
"""
from typing import Tuple

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

    def _get_composition_emb(self, tok_batch: th.LongTensor, lens: th.LongTensor, func: nn.Module) -> th.Tensor:
        h_embs = self.tok_emb(tok_batch)   # (B, L, emb_dim)
        h_embs = pack_padded_sequence(h_embs, lens, batch_first=True, enforce_sorted=False)
        h_embs, (h, c) = func(h_embs)
        h_embs, _ = pad_packed_sequence(h_embs, batch_first=True)  # (B, L, emb_dim)
        h_embs = h_embs[th.arange(h_embs.size(0)), lens-1]  # (B, emb_dim)
        return h_embs

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
        h = F.normalize(h, p=2, dim=1)    # constraint that L2-norm of emb is 1
        r = F.normalize(r, p=2, dim=1)
        t = F.normalize(t, p=2, dim=1)
        score = (h + r - t).norm(p=self.norm, dim=1)   # (batch,)
        return score

    def forward(self, h_batch: th.LongTensor, r_batch: th.LongTensor, t_batch: th.LongTensor,
                h_lens: th.LongTensor, r_lens: th.LongTensor, t_lens: th.LongTensor):
        h_embs = self._get_composition_emb(h_batch, h_lens, self.mention_func)
        r_embs = self._get_composition_emb(r_batch, r_lens, self.rel_func)
        t_embs = self._get_composition_emb(t_batch, t_lens, self.mention_func)
        # embs size = (B, emb_dim)
        # batch negative sampling
        corrupt_h_embs, corrupt_t_embs = self._sample_batch_negative_triples(h_embs, t_embs)
        pos_scores = self._cal_distance(h_embs, r_embs, t_embs)
        neg_scores = self._cal_distance(corrupt_h_embs, r_embs, corrupt_t_embs)
        return pos_scores, neg_scores

    def test_tail_pred(self, h_batch: th.LongTensor, r_batch: th.LongTensor, all_cep_emb: th.FloatTensor,
                       h_lens: th.LongTensor, r_lens: th.LongTensor) -> th.Tensor:
        """
        Args:
            h_batch, t_batch: size = (B, max_l)
            all_cep_emb: size = (cep_cnt, emb_d)
        """
        cep_cnt = all_cep_emb.size(0)
        B = h_batch.size(0)
        h_embs = self._get_composition_emb(h_batch, h_lens, self.mention_func)  # (B, emb_d)
        r_embs = self._get_composition_emb(r_batch, r_lens, self.rel_func)  # (B, emb_d)
        h_embs = h_embs.repeat_interleave(cep_cnt, dim=0)  # (B*cep_cnt, emb_d)
        r_embs = r_embs.repeat_interleave(cep_cnt, dim=0)  # (B*cep_cnt, emb_d)
        all_cep_emb = all_cep_emb.repeat_interleave(B, dim=0)    # (B*cep_cnt, emb_d)
        score = self._cal_distance(h_embs, r_embs, all_cep_emb)   # (B*cep_cnt, )
        return score.reshape(B, cep_cnt)

    def test_head_pred(self, t_batch: th.LongTensor, r_batch: th.LongTensor, all_cep_emb: th.FloatTensor,
                       t_lens: th.LongTensor, r_lens: th.LongTensor) -> th.Tensor:
        """
        Args:
            t_batch, t_batch: size = (B, max_l)
            all_cep_emb: size = (cep_cnt, emb_d)
        """
        cep_cnt = all_cep_emb.size(0)
        B = t_batch.size(0)
        t_embs = self._get_composition_emb(t_batch, t_lens, self.mention_func)  # (B, emb_d)
        r_embs = self._get_composition_emb(r_batch, r_lens, self.rel_func)  # (B, emb_d)
        t_embs = t_embs.repeat_interleave(cep_cnt, dim=0)  # (B*cep_cnt, emb_d)
        r_embs = r_embs.repeat_interleave(cep_cnt, dim=0)  # (B*cep_cnt, emb_d)
        all_cep_emb = all_cep_emb.repeat_interleave(B, dim=0)    # (B*cep_cnt, emb_d)
        score = self._cal_distance(all_cep_emb, r_embs, t_embs)   # (B*cep_cnt, )
        return score.reshape(B, cep_cnt)
