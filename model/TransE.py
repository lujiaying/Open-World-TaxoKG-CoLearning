"""
Adapted from https://github.com/mklimasz/TransE-PyTorch/,
  and Can We Predict New Facts with Open Knowledge Graph Embeddings?
Author: Anonymous Siamese
Create Date: Jul 15, 2021
"""
from typing import Tuple
import random

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchtext

PAD_idx = 0


class BaseModel(nn.Module):
    def __init__(self, tok_count: int, emb_dim: int, norm: int):
        super(BaseModel, self).__init__()
        self.tok_count = tok_count
        self.norm = norm    # norm for (h+r-t), could be 1 or 2
        self.emb_dim = emb_dim
        self.lstm_layer = 1
        # init emb
        uniform_range = 6 / (self.emb_dim**0.5)
        self.tok_emb = nn.Embedding(num_embeddings=self.tok_count,
                                    embedding_dim=self.emb_dim,
                                    padding_idx=PAD_idx)
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
        """
        Args:
            h_embs, t_embs: (B, h)
        """
        B = h_embs.size(0)
        # generate (B, B-1) tensor stores corrputed indices
        corrputed_idices = []
        for i in range(B):
            indices = [_ for _ in range(B) if _ != i]
            corrputed_idices.append(indices)
        corrputed_idices = th.LongTensor(corrputed_idices)  # (B, B-1)
        # whether to corrupt head or tail
        if random.random() >= 0.5:
            # corrupt tail
            corrupt_h_embs = h_embs.repeat_interleave(B-1, dim=0)  # (B*B-1, h)
            corrupt_t_embs = t_embs[corrputed_idices.view(-1, 1).squeeze(1)]  # (B*B-1, h)
        else:
            # corrupt head
            corrupt_h_embs = h_embs[corrputed_idices.view(-1, 1).squeeze(1)]  # (B*B-1, h)
            corrupt_t_embs = t_embs.repeat_interleave(B-1, dim=0)  # (B*B-1, h)
        """
        # change from sample one negative to sample multiple negatives
        batch_size = h_embs.size(0)
        head_or_tail = th.randint(high=2, size=(batch_size,))
        random_rows = th.randperm(batch_size)
        corrupt_hidx = th.where(head_or_tail == 1, random_rows, th.arange(batch_size))
        corrupt_tidx = th.where(head_or_tail == 0, random_rows, th.arange(batch_size))
        corrupt_h_embs = h_embs[corrupt_hidx]
        corrupt_t_embs = t_embs[corrupt_tidx]
        """
        return corrupt_h_embs, corrupt_t_embs

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


class OpenTransE(nn.Module):
    """
    This is the model I implemented first, so not use BaseModel
    """
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
                                    padding_idx=PAD_idx)
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

    def init_tok_emb_by_pretrain(self, tok_vocab: dict, pretrain_name: str):
        if pretrain_name == 'GloVe':
            # emb_dim must in [50, 100, 200, 300]
            pretrain_vecs = torchtext.vocab.GloVe(name='6B', dim=self.emb_dim)
        else:
            print('invalid pretrain_name=%s' % (pretrain_name))
            exit(-1)
        with th.no_grad():
            for tok, idx in tok_vocab.items():
                if idx == PAD_idx:
                    continue
                if tok not in pretrain_vecs.stoi:
                    continue
                vec = pretrain_vecs.get_vecs_by_tokens(tok)
                self.tok_emb.weight[idx] = vec

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


class OpenDistMult(BaseModel):
    def __init__(self, tok_count: int, emb_dim: int, norm: int):
        super(OpenDistMult, self).__init__(tok_count, emb_dim, norm)

    def forward(self, h_batch: th.LongTensor, r_batch: th.LongTensor, t_batch: th.LongTensor,
                h_lens: th.LongTensor, r_lens: th.LongTensor, t_lens: th.LongTensor):
        h_embs = self._get_composition_emb(h_batch, h_lens, self.mention_func)
        r_embs = self._get_composition_emb(r_batch, r_lens, self.rel_func)
        t_embs = self._get_composition_emb(t_batch, t_lens, self.mention_func)
        # embs size = (B, emb_dim)
        # batch negative sampling
        B = h_embs.size(0)
        corrupt_h_embs, corrupt_t_embs = self._sample_batch_negative_triples(h_embs, t_embs)
        pos_scores = self._cal_distance(h_embs, r_embs, t_embs)
        neg_scores = self._cal_distance(corrupt_h_embs, r_embs.repeat_interleave(B-1, dim=0),
                                        corrupt_t_embs)  # (B*B-1, )
        neg_scores = neg_scores.view(B, B-1).mean(dim=1)  # (B, )
        return pos_scores, neg_scores

    def _cal_distance(self, h: th.FloatTensor, r: th.FloatTensor, t: th.FloatTensor) -> th.FloatTensor:
        """
        Args:
            h, r, t: shape=(batch, dim)
        """
        h = F.normalize(h, p=2, dim=1)    # constraint that L2-norm of emb is 1
        r = F.normalize(r, p=2, dim=1)
        t = F.normalize(t, p=2, dim=1)
        score = (h * r * t).sum(dim=1)    # (batch, ), diff from TransE.
        # score the larger the better.
        return score

    def test_tail_pred(self, h_batch: th.LongTensor, r_batch: th.LongTensor, all_cep_emb: th.FloatTensor,
                       h_lens: th.LongTensor, r_lens: th.LongTensor) -> th.Tensor:
        """
        Args:
            h_batch, t_batch: size = (B, max_l)
            all_cep_emb: size = (cep_cnt, emb_d)
        """
        h_embs = self._get_composition_emb(h_batch, h_lens, self.mention_func)  # (B, emb_d)
        r_embs = self._get_composition_emb(r_batch, r_lens, self.rel_func)  # (B, emb_d)
        score = th.mm(h_embs*r_embs, all_cep_emb.transpose(1, 0))  # (B, cep_cnt)
        return score

    def test_head_pred(self, t_batch: th.LongTensor, r_batch: th.LongTensor, all_cep_emb: th.FloatTensor,
                       t_lens: th.LongTensor, r_lens: th.LongTensor) -> th.Tensor:
        """
        Args:
            t_batch, t_batch: size = (B, max_l)
            all_cep_emb: size = (cep_cnt, emb_d)
        """
        t_embs = self._get_composition_emb(t_batch, t_lens, self.mention_func)  # (B, emb_d)
        r_embs = self._get_composition_emb(r_batch, r_lens, self.rel_func)  # (B, emb_d)
        score = th.mm(r_embs*t_embs, all_cep_emb.transpose(1, 0))   # (B, cep_cnt)
        return score


def com_mult(a: th.tensor, b: th.tensor) -> th.tensor:
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return th.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a: th.tensor) -> th.tensor:
    a[..., 1] = -a[..., 1]
    return a


def ccorr(a: th.tensor, b: th.tensor) -> th.tensor:
    """
    Adpated from https://github.com/malllabiisc/CompGCN/blob/master/helper.py
    """
    return th.irfft(com_mult(conj(th.rfft(a, 1)), th.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


class OpenHolE(BaseModel):
    def __init__(self, tok_count: int, emb_dim: int, norm: int):
        super(OpenHolE, self).__init__(tok_count, emb_dim, norm)

    def forward(self, h_batch: th.LongTensor, r_batch: th.LongTensor, t_batch: th.LongTensor,
                h_lens: th.LongTensor, r_lens: th.LongTensor, t_lens: th.LongTensor):
        h_embs = self._get_composition_emb(h_batch, h_lens, self.mention_func)
        r_embs = self._get_composition_emb(r_batch, r_lens, self.rel_func)
        t_embs = self._get_composition_emb(t_batch, t_lens, self.mention_func)
        # embs size = (B, emb_dim)
        # batch negative sampling
        B = h_embs.size(0)
        corrupt_h_embs, corrupt_t_embs = self._sample_batch_negative_triples(h_embs, t_embs)
        pos_scores = self._cal_distance(h_embs, r_embs, t_embs)  # (B, )
        neg_scores = self._cal_distance(corrupt_h_embs, r_embs.repeat_interleave(B-1, dim=0),
                                        corrupt_t_embs)  # (B*B-1, )
        neg_scores = neg_scores.view(B, B-1).mean(dim=1)  # (B, )
        return pos_scores, neg_scores

    def _cal_distance(self, h: th.FloatTensor, r: th.FloatTensor, t: th.FloatTensor) -> th.FloatTensor:
        """
        Args:
            h, r, t: shape=(batch, dim)
        """
        h = F.normalize(h, p=2, dim=1)    # constraint that L2-norm of emb is 1
        r = F.normalize(r, p=2, dim=1)
        t = F.normalize(t, p=2, dim=1)
        score = (r * ccorr(h, t)).sum(dim=1)   # (batch, )
        # score the larger the better.
        return score
